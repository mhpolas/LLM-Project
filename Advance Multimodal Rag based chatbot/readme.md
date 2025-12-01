# 🧠 Multimodal RAG Chat App (Docs + Images)

### 🚀 A Streamlit-powered, history-aware Retrieval-Augmented Generation (RAG) application supporting documents **(PDF, DOCX)** and **images (OCR)** with multiple embedding backends — OpenAI *small/large* and Hugging Face.

---

## 🏗️ Project Overview

This application enables you to **chat with your own documents and images**, powered by **LangChain**, **ChromaDB**, and **OpenAI / Hugging Face** embeddings.

It supports:
- Uploading **PDFs, DOC/DOCX files, and images** (JPG, PNG, TIFF, etc.)
- **OCR-based text extraction** from images using Tesseract
- **History-aware question answering** — ask follow-up questions like ChatGPT
- **Switchable embeddings**:
  - 🧩 OpenAI Small (`text-embedding-3-small`, 1536 dims)
  - 🧠 OpenAI Large (`text-embedding-3-large`, 3072 dims)
  - 🤗 Hugging Face (`all-mpnet-base-v2`, 768 dims)
- **Persistent vector storage** using Chroma (per-provider isolation)
- Full **chat interface** with session memory and historical chat viewing
- Adjustable **temperature** and **OpenAI chat model**

---

## 🧰 Tech Stack

| Category  | Technology |
|-----------|-------------|
| **Frontend / UI** | [Streamlit](https://streamlit.io/) |
| **LLM & Embeddings** | [LangChain 1.x](https://www.langchain.com/), [OpenAI API](https://platform.openai.com/), [Hugging Face Transformers](https://huggingface.co/) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) |
| **OCR** | [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) |
| **File Handling** | PDF, DOC/DOCX, Image, and ZIP upload |
| **Languages** | Python 3.10+ |

---

## 🧩 Key Features

✅ **Multimodal Input Support**  
→ Chat with text extracted from both **documents** and **images** (via OCR).

✅ **Three Embedding Providers**  
→ Choose between **OpenAI Small**, **OpenAI Large**, and **Hugging Face** embeddings in the sidebar UI.

✅ **Chat Memory & History**  
→ Each conversation maintains context and allows **follow-up questions**.  
→ Old chats are preserved and can be reopened anytime.

✅ **RAG Chain (Retrieval-Augmented Generation)**  
→ Uses **LangChain’s history-aware retriever** to rewrite follow-up questions for accurate retrieval.

✅ **Streamlit Interface**  
→ Intuitive ChatGPT-like experience with session control, index rebuilding, and embedding/model switching.

✅ **Persistent Vector Store**  
→ Each embedding model uses a **separate Chroma collection** (`*_openai_small`, `*_openai_large`, `*_huggingface`) to avoid dimension conflicts.



