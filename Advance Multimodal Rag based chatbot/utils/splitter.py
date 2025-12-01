"""
utils/splitter.py  —  LangChain 1.x compatible

Reusable text-splitting utilities for LangChain Document objects.

Provides:
  • split_by_chars(...)  – character-length splitting using RecursiveCharacterTextSplitter
  • split_by_tokens(...) – token-length splitting using TokenTextSplitter

Both functions:
  • Preserve metadata from the original Document(s)
  • Return a flat List[Document] of chunks

Dependencies:
  - langchain-core
  - langchain-text-splitters
"""

from __future__ import annotations

from typing import Iterable, List, Optional

# ✅ LangChain 1.x imports
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def split_by_chars(
    documents: Iterable[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    Split documents by approximate character count (recommended general default).

    Parameters
    ----------
    documents : Iterable[Document]
        Input documents (metadata will be preserved).
    chunk_size : int
        Target maximum characters per chunk. Default: 1000.
    chunk_overlap : int
        Overlap (in characters) between adjacent chunks. Default: 100.
    separators : Optional[List[str]]
        Hierarchical split points. If None, defaults to ["\\n\\n", "\\n", " ", ""].

    Returns
    -------
    List[Document]
        Flattened list of chunked Document objects.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[Document] = []
    for doc in documents:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        chunks.extend(
            splitter.split_documents([doc])  # preserves metadata automatically
        )
    return chunks


def split_by_tokens(
    documents: Iterable[Document],
    *,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 50,
    encoding_name: Optional[str] = None,
) -> List[Document]:
    """
    Split documents by approximate token count (useful for tight LLM context windows).

    Parameters
    ----------
    documents : Iterable[Document]
        Input documents (metadata preserved).
    chunk_size_tokens : int
        Target maximum tokens per chunk. Default: 500.
    chunk_overlap_tokens : int
        Overlap (in tokens) between adjacent chunks. Default: 50.
    encoding_name : Optional[str]
        tiktoken encoding name (e.g., "cl100k_base"). If None, a default is inferred.

    Returns
    -------
    List[Document]
        Flattened list of token-sized chunk Documents.
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        encoding_name=encoding_name,  # None => sensible default
    )

    chunks: List[Document] = []
    for doc in documents:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        parts = splitter.split_text(text)
        meta = dict(doc.metadata) if doc.metadata else {}
        for t in parts:
            chunks.append(Document(page_content=t, metadata=meta))
    return chunks
