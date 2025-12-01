"""
utils/retrieval.py  —  LangChain 1.x compatible

Vector store management + retriever factory for your RAG pipeline.

Features
--------
- Build a Chroma vector store from LangChain Documents
- Optional persistence on disk (re-attach across app restarts)
- Incremental updates (add new documents later)
- Create retrievers with similarity / MMR / score-threshold search
- Optional metadata filters at retrieval time

Dependencies
-----------
- langchain-core
- langchain-community
- chromadb
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

# ✅ LangChain 1.x types & integrations
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma


class VectorStoreManager:
    """
    Thin wrapper around Chroma to keep vector-store concerns isolated from the pipeline.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        *,
        persist_directory: Optional[str] = None,
        collection_name: str = "docs_collection",
    ) -> None:
        """
        Parameters
        ----------
        embedding_model : Embeddings
            Any LangChain-compatible embeddings model (e.g., HuggingFaceEmbeddings).
        persist_directory : Optional[str]
            Directory to store Chroma DB (set None to keep in-memory).
        collection_name : str
            Logical name for the Chroma collection.
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store: Optional[Chroma] = None

    # ----------------------- Build / Load -----------------------

    def build_from_documents(self, documents: List[Document]) -> None:
        """
        Create (or overwrite) a Chroma collection from the given documents.
        """
        if not documents:
            raise ValueError("No documents provided to build the vector store.")

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,          # ✅ correct kw for 1.x
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )

        if self.persist_directory:
            # In newer Chroma versions persistence is automatic when persist_directory is set,
            # but calling persist() is safe and explicit.
            self.vector_store.persist()

    def load_existing(self) -> None:
        """
        Re-attach to an existing persisted Chroma collection (without rebuilding).
        """
        if not self.persist_directory:
            raise ValueError("persist_directory is not set; cannot load existing collection.")

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,  # ✅ correct kw for loading path
            persist_directory=self.persist_directory,
        )

    # ----------------------- Incremental Updates -----------------------

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to an existing store (builds from scratch if not yet built).
        """
        if not documents:
            return

        if self.vector_store is None:
            self.build_from_documents(documents)
            return

        self.vector_store.add_documents(documents)
        if self.persist_directory:
            self.vector_store.persist()

    # ----------------------- Retriever Factory -----------------------

    def get_retriever(
        self,
        *,
        k: int = 4,
        search_type: str = "similarity",                 # "similarity" | "mmr" | "similarity_score_threshold"
        fetch_k: Optional[int] = None,                   # for "mmr"
        score_threshold: Optional[float] = None,         # for "similarity_score_threshold"
        metadata_filter: Optional[Dict[str, Any]] = None # Chroma metadata filter
    ) -> BaseRetriever:
        """
        Create a retriever from the current vector store.

        Parameters
        ----------
        k : int
            Number of documents to return.
        search_type : str
            Retrieval mode: "similarity", "mmr", or "similarity_score_threshold".
        fetch_k : Optional[int]
            For MMR: number of candidates to fetch before diversity re-ranking.
        score_threshold : Optional[float]
            For score-threshold search: minimum similarity score (0..1).
        metadata_filter : Optional[Dict[str, Any]]
            Chroma metadata filter, e.g., {"modality": "pdf"} or {"source": {"$in": ["a.pdf","b.pdf"]}}.

        Returns
        -------
        BaseRetriever
            A LangChain retriever object for use in RetrievalQA or custom chains.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Build or load it first.")

        search_kwargs: Dict[str, Any] = {"k": k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        if search_type == "mmr":
            if fetch_k:
                search_kwargs["fetch_k"] = fetch_k
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs,
            )
        elif search_type == "similarity_score_threshold":
            if score_threshold is None:
                raise ValueError("Provide score_threshold for 'similarity_score_threshold' search_type.")
            search_kwargs["score_threshold"] = float(score_threshold)
            retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs,
            )
        else:
            # Default: standard similarity search
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs,
            )

        return retriever
