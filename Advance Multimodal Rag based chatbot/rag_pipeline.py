"""
rag_pipeline.py — LangChain 1.x compatible, history-aware chat RAG

Adds:
  - Configurable model & temperature (via init or env)
  - History-aware retriever (follow-up questions use prior turns)
  - RunnableWithMessageHistory to persist chat memory per session_id
  - Grounded prompt with {context}, {input}, and {chat_history}
  - Sources preserved for UI

Env (optional)
--------------
OPENAI_API_KEY
OPENAI_MODEL_NAME                  # e.g., "gpt-3.5-turbo" (default) or "gpt-4o"
EMBEDDING_MODEL                    # e.g., "sentence-transformers/all-mpnet-base-v2"
CHROMA_PERSIST_DIRECTORY           # e.g., "vector_store"
CHROMA_COLLECTION_NAME             # e.g., "docs_collection"
"""

from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

# Core LC 1.x
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Providers
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Chains (modern constructors)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

# Chat history store
from langchain_community.chat_message_histories import ChatMessageHistory

from utils.retrieval import VectorStoreManager


class RAGPipeline:
    """
    Chat-ready, history-aware RAG pipeline.

    Use:
      pipe = RAGPipeline(embedding_provider="openai", openai_embedding_model="text-embedding-3-small")
      pipe.build(documents)
      ans = pipe.ask("Follow-up?", session_id="abc123")
      srcs = pipe.get_last_sources()
    """

    def __init__(
        self,
        *,
        embedding_provider: str = "openai",            # "openai" | "huggingface"
        embedding_model_name: Optional[str] = None,    # HF model name
        openai_embedding_model: Optional[str] = None,  # "text-embedding-3-small" | "text-embedding-3-large"
        llm_model_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        # Retrieval knobs
        k: int = 4,
        search_type: str = "similarity",               # "similarity" | "mmr" | "similarity_score_threshold"
        fetch_k: Optional[int] = None,                 # for "mmr"
        score_threshold: Optional[float] = None,       # for "similarity_score_threshold"
        metadata_filter: Optional[Dict[str, Any]] = None,
        # LLM behavior
        temperature: float = 0.0,
    ) -> None:
        load_dotenv()

        # -------- Config (env-backed defaults) --------
        self.embedding_provider = (embedding_provider or "openai").lower()

        self.embedding_model_name = (
            embedding_model_name
            or os.getenv("EMBEDDING_MODEL")
            or "sentence-transformers/all-mpnet-base-v2"
        )
        self.openai_embedding_model = (
            openai_embedding_model
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )

        self.llm_model_name = (
            llm_model_name
            or os.getenv("OPENAI_MODEL_NAME")
            or "gpt-3.5-turbo"
        )
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY") or None
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME") or "docs_collection"

        # -------- Embeddings & Vector Store --------
        self.embeddings = self._init_embeddings()
        self.vs_manager = VectorStoreManager(
            embedding_model=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        # -------- LLM (requires OPENAI_API_KEY) --------
        self.llm = ChatOpenAI(model=self.llm_model_name, temperature=float(temperature))

        # -------- Retrieval settings --------
        self.k = int(k)
        self.search_type = search_type
        self.fetch_k = fetch_k
        self.score_threshold = score_threshold
        self.metadata_filter = metadata_filter

        # -------- Prompts --------
        # 1) Rewriter prompt: turn follow-ups into standalone queries using chat_history
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a query rewriter. Given a conversation and a follow-up question, "
                 "rewrite the question to be a standalone query for searching a knowledge base. "
                 "Keep it concise and preserve important details."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 2) Answer prompt: ground answers in retrieved context + chat history
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a careful assistant. Answer strictly from the provided context. "
                 "If the answer is not in the context, say you don't know.\n"
                 "Use the conversation history to maintain continuity and resolve references."),
                MessagesPlaceholder("chat_history"),
                ("system", "Context:\n{context}"),
                ("human", "Question:\n{input}"),
            ]
        )

        # -------- Chains / Memory --------
        self._with_history = None          # RunnableWithMessageHistory wrapper
        self._last_sources: List[Document] = []
        self._memory_store: Dict[str, ChatMessageHistory] = {}

    # ---------------- Internal: Embeddings ----------------

    def _init_embeddings(self):
        """Initialize embedding model according to provider setting."""
        provider = self.embedding_provider.lower()
        if provider == "huggingface":
            print(f"🔤 Using Hugging Face embeddings: {self.embedding_model_name}")
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        else:
            # openai provider
            print(f"🔤 Using OpenAI embeddings: {self.openai_embedding_model}")
            return OpenAIEmbeddings(model=self.openai_embedding_model)

    def set_embedding_provider(
        self,
        provider: str,
        *,
        openai_model: Optional[str] = None,
        hf_model: Optional[str] = None,
    ) -> None:
        """Change embedding backend dynamically; call build() again to rewire chains."""
        self.embedding_provider = (provider or "openai").lower()
        if openai_model:
            self.openai_embedding_model = openai_model
        if hf_model:
            self.embedding_model_name = hf_model
        self.embeddings = self._init_embeddings()
        # Re-link VS manager with new embedder (collection name can be changed by caller/ UI)
        self.vs_manager = VectorStoreManager(
            embedding_model=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        self._with_history = None

    def set_model(self, *, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        """Change chat model/temperature; call build() again to rewire chains."""
        if model:
            self.llm_model_name = model
        if temperature is not None:
            self.llm = ChatOpenAI(model=self.llm_model_name, temperature=float(temperature))
        self._with_history = None

    # ---------------- Build / Rebuild ----------------

    def build(self, *, documents: List[Document]) -> None:
        """
        Build the vector store and create a history-aware retrieval chain.
        """
        if not documents:
            raise ValueError("No documents provided to build the RAG pipeline.")

        # Build Chroma store
        self.vs_manager.build_from_documents(documents)

        # Base retriever
        base_retriever = self.vs_manager.get_retriever(
            k=self.k,
            search_type=self.search_type,
            fetch_k=self.fetch_k,
            score_threshold=self.score_threshold,
            metadata_filter=self.metadata_filter,
        )

        # 1) History-aware retriever (rewrites follow-ups → standalone search query)
        hist_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=base_retriever,
            prompt=self.rewrite_prompt,
        )

        # 2) Combine-docs (stuff) chain to answer with context + chat_history
        stuff_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.answer_prompt,
        )

        # 3) Retrieval chain (history-aware retriever → answer chain)
        retrieval_chain = create_retrieval_chain(
            retriever=hist_retriever,
            combine_docs_chain=stuff_chain,
            #return_source_documents=True,
        )

        # 4) Add message history wrapper so we pass chat_history seamlessly each call
        def _get_history(session_id: str) -> ChatMessageHistory:
            if session_id not in self._memory_store:
                self._memory_store[session_id] = ChatMessageHistory()
            return self._memory_store[session_id]

        self._with_history = RunnableWithMessageHistory(
            retrieval_chain,
            _get_history,
            input_messages_key="input",     # user question
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    # ---------------- Inference ----------------

    def ask(self, question: str, *, session_id: str = "default") -> str:
        """
        Ask a question with chat memory (session_id). Returns an answer string.
        Sources from the last call are available via get_last_sources().
        """
        if self._with_history is None:
            raise RuntimeError("Pipeline not built. Call .build(documents=...) first.")

        result = self._with_history.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        answer = result.get("answer", result.get("result", ""))
        self._last_sources = result.get("source_documents", []) or []
        return answer

    def get_last_sources(self) -> List[Document]:
        """Source documents used in the last call to ask()."""
        return self._last_sources

    # ---------------- Utilities ----------------

    def reset_session(self, session_id: str = "default") -> None:
        """Clear chat history for a session (without rebuilding the index)."""
        if session_id in self._memory_store:
            self._memory_store[session_id].clear()