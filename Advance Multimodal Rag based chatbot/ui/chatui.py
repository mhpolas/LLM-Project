# ui/chat_app.py
"""
Chat-style Streamlit UI (ChatGPT-like) for history-aware RAG over uploaded files.

Highlights
----------
- Upload DOC/DOCX, PDF, images (or ZIP of these)
- Build a Chroma index and query with a history-aware retrieval chain
- Sidebar controls for LLM model, temperature, and Embeddings:
    • OpenAI (small) → text-embedding-3-small
    • OpenAI (large) → text-embedding-3-large
    • Hugging Face   → sentence-transformers/all-mpnet-base-v2
- True chat memory (follow-ups use prior turns)
- View & switch "Old chat history"
- Safe per-embedding-model collection naming to avoid Chroma dimension clashes

Run:
  python -m streamlit run ui/chat_app.py
"""

from __future__ import annotations

import os
import uuid
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

from utils.docs_loader import load_documents_from_files
from rag_pipeline import RAGPipeline


# --------------------------- Session State Helpers ---------------------------

def _ensure_state():
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = None
    if "docs" not in st.session_state:
        st.session_state["docs"] = []             # chunked docs used to build the index
    if "docs_count" not in st.session_state:
        st.session_state["docs_count"] = 0
    if "sessions" not in st.session_state:
        st.session_state["sessions"] = {}         # {session_id: [{"role":..., "content":..., "sources":[...]}, ...]}
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = str(uuid.uuid4())


def _messages() -> List[Dict[str, Any]]:
    sid = st.session_state["current_session_id"]
    return st.session_state["sessions"].setdefault(sid, [])


def _switch_session(new_sid: str):
    if new_sid not in st.session_state["sessions"]:
        st.session_state["sessions"][new_sid] = []
    st.session_state["current_session_id"] = new_sid


def _source_summaries(docs: List[Document], max_chars: int = 600) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs:
        meta = d.metadata or {}
        text = (d.page_content or "").strip()
        out.append({
            "source": meta.get("source", "unknown"),
            "modality": meta.get("modality", "unknown"),
            "filetype": meta.get("filetype", ""),
            "snippet": (text[:max_chars] + "…") if len(text) > max_chars else text,
        })
    return out


def _render_sources(summaries: List[Dict[str, Any]]):
    if not summaries:
        return
    with st.expander("📎 Sources used", expanded=False):
        for i, s in enumerate(summaries, start=1):
            label = f"{i}. {s['source']} · {s['modality']}"
            if s.get("filetype"):
                label += f" ({s['filetype']})"
            st.markdown(f"**{label}**")
            st.write(s["snippet"] or "*No preview available*")
            st.markdown("---")


def _compute_collection_name(base_name: str, provider: str, openai_embed_model: str, hf_model: str) -> str:
    """
    Create a dimension-safe collection name: base_provider_modelslug
    """
    if provider == "openai":
        slug = openai_embed_model.replace("-", "_")
    else:
        slug = hf_model.replace("-", "_").replace("/", "_")
    return f"{base_name}_{provider}_{slug}"


# --------------------------- App ---------------------------

def main():
    load_dotenv()
    _ensure_state()

    st.set_page_config(page_title="Chat with your Docs (History-aware RAG)", layout="wide")
    st.title("💬 Chat with your Documents (Multimodal RAG + History)")

    # ---------------- Sidebar: Controls ----------------
    with st.sidebar:
        st.header("Settings")

        # --- Embedding choice: three-way ---
        embed_option = st.selectbox(
            "Embeddings",
            ["OpenAI (small)", "OpenAI (large)", "Hugging Face"],
            index=0,
            help="Choose embeddings for retrieval. Collection name auto-suffixed to avoid dimension clashes.",
        )
        if embed_option.startswith("OpenAI"):
            embedding_provider = "openai"
            openai_embed_model = "text-embedding-3-small" if "small" in embed_option else "text-embedding-3-large"
            # Keep env in sync (optional)
            os.environ["OPENAI_EMBEDDING_MODEL"] = openai_embed_model
            hf_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        else:
            embedding_provider = "huggingface"
            openai_embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            hf_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

        # Chat model + temperature
        default_model = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        model_name = st.text_input("OpenAI Chat Model", value=default_model, help="e.g., gpt-3.5-turbo, gpt-4o")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        # Vector store persistence
        persist_dir = st.text_input(
            "Chroma persist directory (optional)",
            value=os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip(),
            help="Leave empty for in-memory (rebuilt each run)."
        )
        base_collection = st.text_input(
            "Base collection name",
            value=os.getenv("CHROMA_COLLECTION_NAME", "docs_collection"),
        )
        computed_collection = _compute_collection_name(base_collection, embedding_provider, openai_embed_model, hf_model)

        # Chunking
        chunk_size = st.number_input("Chunk size (chars)", min_value=100, max_value=6000, value=1000, step=100)
        chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=2000, value=100, step=50)

        # Retrieval
        search_type = st.selectbox("Search type", ["similarity", "mmr", "similarity_score_threshold"], index=0)
        k = st.number_input("k (top results)", min_value=1, max_value=20, value=4, step=1)
        fetch_k = None
        score_threshold = None
        if search_type == "mmr":
            fetch_k = st.number_input("fetch_k (MMR candidates)", min_value=1, max_value=100, value=20, step=1)
        elif search_type == "similarity_score_threshold":
            score_threshold = st.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # OCR
        ocr_lang = st.text_input(
            "OCR language(s)",
            value=os.getenv("OCR_LANG", "eng"),
            help='Tesseract codes, e.g., "eng", "eng+deu". Ensure language packs are installed.'
        )

        st.markdown("---")
        # Session/history controls
        st.subheader("Chat Sessions")
        sessions = st.session_state["sessions"]
        if sessions:
            options = list(sessions.keys())
            labels = [f"{i+1}. {sid[:8]}… ({len(sessions[sid])} msgs)" for i, sid in enumerate(options)]
            selected = st.selectbox("Old chat history", options=options, format_func=lambda sid: labels[options.index(sid)])
            if selected and selected != st.session_state["current_session_id"]:
                _switch_session(selected)
                st.success(f"Switched to chat: {selected[:8]}…")
        else:
            st.caption("No previous chats yet.")

        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("🆕 New Chat"):
                new_id = str(uuid.uuid4())
                _switch_session(new_id)
                st.success(f"Created new chat: {new_id[:8]}…")
        with colB:
            if st.button("🗑️ Clear Current Chat"):
                st.session_state["sessions"][st.session_state["current_session_id"]] = []
                pipe = st.session_state.get("pipeline")
                if pipe:
                    pipe.reset_session(st.session_state["current_session_id"])
                st.success("Cleared current chat history.")
        with colC:
            if st.button("✅ Apply Model/Embeddings"):
                pipe: RAGPipeline | None = st.session_state.get("pipeline")
                docs = st.session_state.get("docs", [])
                if pipe and docs:
                    # Model & temperature
                    pipe.set_model(model=model_name, temperature=temperature)
                    # Embeddings
                    pipe.set_embedding_provider(
                        embedding_provider,
                        openai_model=openai_embed_model,
                        hf_model=hf_model,
                    )
                    # Change collection to dimension-safe name
                    pipe.collection_name = computed_collection
                    # Rebuild chains with existing docs
                    pipe.build(documents=docs)
                    st.success("Applied model/temperature/embeddings.")
                else:
                    st.info("Settings saved. Build the pipeline after uploading files.")

        st.caption(
            f"Active collection: `{computed_collection}`"
        )

    # ---------------- Upload & Build ----------------
    st.subheader("📁 Upload & Build")
    uploads = st.file_uploader(
        "Upload documents (or a ZIP of them)",
        type=["doc", "docx", "pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "zip"],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        build_clicked = st.button("⚙️ Build / Rebuild Index")
    with col2:
        reset_index = st.button("🧹 Reset Index")

    if reset_index:
        st.session_state["pipeline"] = None
        st.session_state["docs"] = []
        st.session_state["docs_count"] = 0
        st.success("Index cleared. You can upload and build again.")
        st.stop()

    if build_clicked:
        if not uploads:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Loading & chunking…"):
                try:
                    docs = load_documents_from_files(
                        uploads,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        ocr_lang=ocr_lang,
                    )
                except Exception as e:
                    st.error(f"Failed to load documents: {e}")
                    docs = []

            st.session_state["docs"] = docs
            st.session_state["docs_count"] = len(docs)
            st.info(f"Prepared **{len(docs)}** chunk(s).")

            if docs:
                with st.spinner("Building vector store + history-aware retrieval…"):
                    pipe = RAGPipeline(
                        embedding_provider=embedding_provider,
                        openai_embedding_model=openai_embed_model,
                        embedding_model_name=hf_model,
                        persist_directory=(persist_dir or None),
                        collection_name=computed_collection,  # dimension-safe
                        k=int(k),
                        search_type=search_type,
                        fetch_k=int(fetch_k) if fetch_k else None,
                        score_threshold=float(score_threshold) if score_threshold is not None else None,
                        temperature=float(temperature),
                        llm_model_name=model_name,
                    )
                    pipe.build(documents=docs)
                    st.session_state["pipeline"] = pipe
                st.success("✅ Ready! Start chatting below.")
            else:
                st.warning("No chunks created. If you uploaded images, verify Tesseract OCR is installed and working.")

    # ---------------- Chat UI ----------------
    st.markdown("---")
    st.subheader("💬 Chat")

    # Render existing messages for current session
    for msg in _messages():
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # Chat input
    user_text = st.chat_input("Ask a question about your uploaded documents…")
    if user_text:
        _messages().append({"role": "user", "content": user_text})

        pipe: RAGPipeline | None = st.session_state.get("pipeline")
        if not pipe:
            with st.chat_message("assistant"):
                st.warning("Please upload files and click **Build / Rebuild Index** first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        sid = st.session_state["current_session_id"]
                        answer = pipe.ask(user_text, session_id=sid)
                        sources = _source_summaries(pipe.get_last_sources())
                        st.write(answer)
                        _render_sources(sources)
                        _messages().append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except Exception as e:
                        err = f"Failed to generate answer: {e}"
                        st.error(err)
                        _messages().append(
                            {"role": "assistant", "content": err, "sources": []}
                        )

    st.caption(
        f"🔎 Indexed chunks: {st.session_state.get('docs_count', 0)}  ·  "
        f"Current chat: {st.session_state['current_session_id'][:8]}…  ·  "
        f"Collection: {computed_collection}"
    )


if __name__ == "__main__":
    main()
