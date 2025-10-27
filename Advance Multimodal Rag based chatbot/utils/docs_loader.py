"""
utils/docs_loader.py  (LangChain 1.x ready)

Unified loader for:
  • Word docs: .docx / .doc
  • PDFs:      .pdf
  • Images:    .png, .jpg, .jpeg, .webp, .tif, .tiff
  • ZIPs:      archives containing any of the above

Outputs:
  • List[langchain_core.documents.Document] already CHUNKED with metadata:
      - metadata["source"]   -> original filename
      - metadata["modality"] -> "document" | "pdf" | "image"
      - metadata["filetype"] -> extension (e.g., ".pdf")

Dependencies:
  - langchain-core
  - langchain-community
  - langchain-text-splitters
  - pillow
  - pytesseract
  - unstructured[docx,pdf]
  - pdfminer.six
  - pypdf (fallback)
System:
  - Tesseract OCR binary must be installed (set TESSERACT_CMD if not on PATH).
"""

from __future__ import annotations

import io
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Iterable, List, Union, Optional

from PIL import Image, ImageOps
import pytesseract

# ✅ LangChain 1.x imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ Community loaders (1.x)
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
try:
    # Fallback PDF loader if Unstructured has issues in your env
    from langchain_community.document_loaders import PyPDFLoader
    _HAS_PYPDF = True
except Exception:
    _HAS_PYPDF = False


# ---- Supported extensions ----
DOC_EXTS    = {".docx", ".doc"}
PDF_EXTS    = {".pdf"}
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
ZIP_EXTS    = {".zip"}
SUPPORTED   = DOC_EXTS | PDF_EXTS | IMAGE_EXTS | ZIP_EXTS

# ---- OCR config (optional) ----
_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_documents_from_files(
    uploaded_files: List[Union[io.BytesIO, Path]],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    ocr_lang: str = "eng",
    pdf_use_unstructured: bool = True,
) -> List[Document]:
    """
    Load mixed files (DOCX/PDF/Images/ZIP), extract text (OCR for images),
    and return CHUNKED LangChain Documents.

    Parameters
    ----------
    uploaded_files : list of file-like objects (e.g., Streamlit UploadedFile) or Path
    chunk_size     : max characters per chunk (default 1000)
    chunk_overlap  : overlapping chars between chunks (default 100)
    ocr_lang       : Tesseract language(s), e.g., "eng" or "eng+deu"
    pdf_use_unstructured : if True, prefer UnstructuredPDFLoader; else try PyPDFLoader

    Returns
    -------
    List[Document] : chunked documents with metadata (source/modality/filetype)
    """
    raw_docs: List[Document] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1) Save all inputs into temp dir; expand ZIPs
        for path in _materialize_to_temp(uploaded_files, tmpdir_path):
            ext = path.suffix.lower()

            if ext in DOC_EXTS:
                raw_docs.extend(_load_word(path))

            elif ext in PDF_EXTS:
                raw_docs.extend(_load_pdf(path, use_unstructured=pdf_use_unstructured))

            elif ext in IMAGE_EXTS:
                img_doc = _load_image(path, ocr_lang=ocr_lang)
                if img_doc:
                    raw_docs.append(img_doc)

            # ZIPs are already expanded in _materialize_to_temp()

    # 2) Chunk uniformly for embeddings/RAG
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],  # good general-purpose defaults
    )

    chunked_docs: List[Document] = []
    for d in raw_docs:
        if not (d.page_content and d.page_content.strip()):
            continue
        chunked_docs.extend(splitter.split_documents([d]))

    return chunked_docs


# ------------------ Helpers: materialize inputs ------------------

def _materialize_to_temp(
    uploaded_files: List[Union[io.BytesIO, Path]],
    tmpdir: Path,
) -> Iterable[Path]:
    """
    Write all inputs to temp dir, expand ZIPs, and yield file paths to process.
    """
    for f in uploaded_files:
        name = getattr(f, "name", None)
        if not name and isinstance(f, Path):
            name = f.name
        if not name:
            name = "uploaded_file"

        # Save in-memory files to disk; copy path-like otherwise
        if hasattr(f, "read"):
            dest = tmpdir / Path(name).name
            with open(dest, "wb") as out:
                out.write(f.read())
        else:
            src = Path(f)
            dest = tmpdir / src.name
            if src.resolve() != dest.resolve():
                dest.write_bytes(src.read_bytes())

        ext = dest.suffix.lower()
        if ext in ZIP_EXTS:
            _extract_zip(dest, tmpdir)
            for p in tmpdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in (DOC_EXTS | PDF_EXTS | IMAGE_EXTS):
                    yield p
        elif ext in (DOC_EXTS | PDF_EXTS | IMAGE_EXTS):
            yield dest
        else:
            # unsupported -> skip
            continue


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    except Exception:
        # ignore malformed archives
        pass


# ------------------ Modality loaders ------------------

def _load_word(path: Path) -> List[Document]:
    loader = UnstructuredWordDocumentLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["modality"] = "document"
        d.metadata["filetype"] = path.suffix.lower()
    return docs


def _load_pdf(path: Path, *, use_unstructured: bool = True) -> List[Document]:
    docs: List[Document] = []
    if use_unstructured:
        try:
            loader = UnstructuredPDFLoader(str(path))
            docs = loader.load()
        except Exception:
            if _HAS_PYPDF:
                loader = PyPDFLoader(str(path))
                docs = loader.load()
            else:
                raise
    else:
        if _HAS_PYPDF:
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        else:
            loader = UnstructuredPDFLoader(str(path))
            docs = loader.load()

    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["modality"] = "pdf"
        d.metadata["filetype"] = path.suffix.lower()
    return docs


def _load_image(path: Path, ocr_lang: str = "eng") -> Optional[Document]:
    try:
        with Image.open(path) as im:
            text = _ocr_image(im, ocr_lang=ocr_lang)
        if text.strip():
            return Document(
                page_content=text.strip(),
                metadata={
                    "source": path.name,
                    "modality": "image",
                    "filetype": path.suffix.lower(),
                },
            )
    except Exception:
        return None
    return None


# ------------------ OCR ------------------

def _ocr_image(im: Image.Image, ocr_lang: str = "eng") -> str:
    """
    Minimal preprocessing + Tesseract OCR.
    """
    img = ImageOps.grayscale(im.convert("RGB"))
    img = ImageOps.autocontrast(img)

    # Upscale small images to help OCR
    min_dim = min(img.size)
    if min_dim < 600:
        scale = max(2, int(600 / max(1, min_dim)))
        img = img.resize((img.size[0] * scale, img.size[1] * scale))

    config = "--oem 3 --psm 6"
    try:
        text = pytesseract.image_to_string(img, lang=ocr_lang, config=config)
    except Exception:
        return ""
    return text.strip()
