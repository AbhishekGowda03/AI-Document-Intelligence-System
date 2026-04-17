import os

# Cap thread counts before any ML library loads to prevent memory explosion
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from services.pdf_parser import extract_text_from_pdf
from services.cleaner import clean_text
from services.chunker import chunk_text
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore

# Set page config early
st.set_page_config(page_title="AI Document Intelligence", page_icon="🧠", layout="wide")

# Custom CSS for modern design
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .stSidebar {
        background-color: #1e293b;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .stButton>button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        transition: transform 0.2s;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

from services.retrieval import RetrievalService

# Load only the lightweight models at startup.
# QA, Summarizer, and Reranker load lazily on first use to avoid OOM.
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embeddings():
    return EmbeddingService()

@st.cache_resource(show_spinner="Loading Vector Store...")
def load_vector_store():
    return VectorStore()

@st.cache_resource
def load_retrieval_service(_vs, _emb):
    from services.retrieval import RetrievalService
    return RetrievalService(_vs, _emb)

emb_service = load_embeddings()
vector_store = load_vector_store()
retrieval_service = load_retrieval_service(vector_store, emb_service)

with st.sidebar:
    st.markdown("## 🧠 Knowledge Engine")
    st.markdown("Upload PDFs to inject external knowledge into the system.")
    
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Index Documents") and uploaded_files:
        with st.spinner("Extracting, Cleaning, Chunking & Embedding..."):
            vector_store.clear()
            total_chunks = 0
            for file in uploaded_files:
                raw_text = extract_text_from_pdf(file.read())
                clean = clean_text(raw_text)
                chunks = chunk_text(clean)
                
                if not chunks:
                    continue
                    
                embeddings = emb_service.get_embeddings(chunks)
                vector_store.add_embeddings(embeddings, chunks)
                total_chunks += len(chunks)
                
            st.success(f"Indexed {len(uploaded_files)} files into {total_chunks} embeddings.")

st.title("AI Document Intelligence")
st.markdown("Dynamic RAG Engine utilizing precision Extractive QA and Abstractive T5 Summarization.")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            st.caption(msg["meta"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            response = retrieval_service.retrieve_and_answer(prompt)
            
            answer = response["answer"]
            route = response["type"].upper()
            latency = response["latency_ms"]
            
            st.markdown(answer)
            meta_str = f"⚡ Latency: {latency}ms | 🧠 Selected Route: {route}"
            st.caption(meta_str)
            
            with st.expander("View Source Chunks", expanded=False):
                for idx, src in enumerate(response["sources"], 1):
                    st.markdown(f"**Chunk {idx}:**\n> {src}\n")
                    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": meta_str
    })
