# ui/streamlit_app.py
"""
Streamlit UI for the Multimodal RAG app (DOCX + PDF + Images)

Features
--------
- Single uploader for .docx/.doc, .pdf, images (.png/.jpg/.jpeg/.webp/.tif/.tiff), and .zip bundles
- Sidebar settings for chunking, retrieval, persistence, and OCR language
- Build vector store & pipeline from uploaded files
- Ask questions; display answers and (optional) source snippets/provenance

Run locally:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

from utils.docs_loader import load_documents_from_files
from rag_pipeline import RAGPipeline


# --------------------------- Helpers ---------------------------

def _render_sources(sources: List[Document], max_chars: int = 600) -> None:
    """
    Nicely render the provenance/source snippets for the last answer.
    """
    if not sources:
        return

    st.markdown("#### Sources used")
    for i, doc in enumerate(sources, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        modality = meta.get("modality", "unknown")
        filetype = meta.get("filetype", "")
        text = doc.page_content or ""
        snippet = (text[:max_chars] + "…") if len(text) > max_chars else text

        with st.expander(f"Source {i}: {source}  ·  {modality}{' (' + filetype + ')' if filetype else ''}", expanded=False):
            st.write(snippet)


def _init_session_state():
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = None
    if "docs_count" not in st.session_state:
        st.session_state["docs_count"] = 0


# --------------------------- App ---------------------------

def main():
    load_dotenv()  # Load .env if present (OPENAI_API_KEY, etc.)
    _init_session_state()

    st.set_page_config(page_title="Multimodal RAG: DOCX + PDF + Images", layout="wide")
    st.title("📚🔎 Multimodal Document Q&A (RAG + OpenAI GPT)")

    # ---------------- Sidebar: Settings ----------------
    with st.sidebar:
        st.header("Settings")

        # Vector store persistence
        persist_dir = st.text_input(
            "Chroma persist directory (optional)",
            value=os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip(),
            help="Leave empty for in-memory (rebuilds each run).",
        )
        collection_name = st.text_input(
            "Collection name",
            value=os.getenv("CHROMA_COLLECTION_NAME", "docs_collection"),
        )

        # Chunking
        chunk_size = st.number_input("Chunk size (chars)", min_value=100, max_value=6000, value=1000, step=100)
        chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=2000, value=100, step=50)

        # Retrieval
        search_type = st.selectbox("Search type", options=["similarity", "mmr", "similarity_score_threshold"], index=0)
        k = st.number_input("k (top results)", min_value=1, max_value=20, value=4, step=1)
        fetch_k = None
        score_threshold = None
        if search_type == "mmr":
            fetch_k_val = st.number_input("fetch_k (MMR candidates)", min_value=1, max_value=100, value=20, step=1)
            fetch_k = int(fetch_k_val)
        elif search_type == "similarity_score_threshold":
            score_threshold = st.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # OCR
        ocr_lang = st.text_input(
            "OCR language(s)",
            value=os.getenv("OCR_LANG", "eng"),
            help='Tesseract codes, e.g., "eng", "eng+deu". Requires Tesseract on the host.',
        )

        st.markdown("---")
        st.caption("Supported: .docx/.doc · .pdf · .png/.jpg/.jpeg/.webp/.tif/.tiff · .zip of any of these.")

    # --------------- Uploader ---------------
    uploaded_files = st.file_uploader(
        "Upload documents (DOCX/PDF/images) or a ZIP containing them",
        type=["docx", "doc", "pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "zip"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.expander("Uploaded files", expanded=False):
            for uf in uploaded_files:
                st.write("•", uf.name)

        # Load & chunk
        with st.spinner("🧩 Loading and chunking…"):
            try:
                chunked_docs = load_documents_from_files(
                    uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    ocr_lang=ocr_lang,
                )
                st.session_state["docs_count"] = len(chunked_docs)
            except Exception as e:
                st.error(f"Failed to load documents: {e}")
                chunked_docs = []

        st.success(f"Prepared {len(chunked_docs)} chunk(s).")

        # Build pipeline
        if st.button("⚙️ Build vector store & pipeline"):
            if not chunked_docs:
                st.warning("No chunked documents available.")
            else:
                with st.spinner("🏗️ Building vector store…"):
                    pipeline = RAGPipeline(
                        persist_directory=(persist_dir or None),
                        collection_name=collection_name,
                        k=int(k),
                        search_type=search_type,
                        fetch_k=fetch_k,
                        score_threshold=score_threshold,
                        temperature=0.0,
                    )
                    pipeline.build(documents=chunked_docs)
                st.session_state["pipeline"] = pipeline
                st.success("✅ Pipeline ready. Ask questions below.")

    # --------------- Q&A ---------------
    pipeline: RAGPipeline | None = st.session_state.get("pipeline")
    if pipeline:
        st.markdown("### 💬 Ask a question")
        question = st.text_input("Your question:")
        ask_clicked = st.button("Ask")

        if ask_clicked:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking…"):
                    try:
                        answer = pipeline.ask(question)
                        st.markdown("#### Answer")
                        st.write(answer)

                        # Show provenance
                        _render_sources(pipeline.get_last_sources())
                    except Exception as e:
                        st.error(f"Failed to generate answer: {e}")

    # --------------- Footer / Reset ---------------
    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        if st.button("🔄 Reset (clear pipeline)"):
            st.session_state["pipeline"] = None
            st.session_state["docs_count"] = 0
            st.rerun()
    with cols[1]:
        st.caption(f"Chunks indexed: {st.session_state.get('docs_count', 0)}")
    with cols[2]:
        st.caption("Tip: Use persistence to avoid rebuilding on restarts.")


if __name__ == "__main__":
    main()
