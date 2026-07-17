import os
import tempfile
import streamlit as st

from src.loaders import load_uploaded_files
from src.rag_pipeline import build_vectorstore, answer_question

st.set_page_config(page_title="RAG Document QA", page_icon="📚", layout="wide")
st.title("📚 RAG-Based Document Question Answering System")

st.markdown(
    "Upload one or more files (PDF, DOCX, TXT, MD, CSV) and ask questions grounded in your documents."
)

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-k retrieved chunks", min_value=2, max_value=8, value=4, step=1)
    chunk_size = st.slider("Chunk size", min_value=400, max_value=1500, value=800, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=400, value=150, step=25)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "md", "csv"],
    accept_multiple_files=True
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if uploaded_files:
    if st.button("Build / Rebuild Index"):
        with st.spinner("Loading and indexing documents..."):
            docs = load_uploaded_files(uploaded_files)
            if not docs:
                st.error("No readable content found in uploaded files.")
            else:
                st.session_state.vector_db = build_vectorstore(
                    docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                st.success(f"Indexed {len(docs)} document units successfully.")

question = st.text_input("Ask a question about your uploaded documents:")

if question:
    if st.session_state.vector_db is None:
        st.warning("Please upload files and click 'Build / Rebuild Index' first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            result = answer_question(
                vector_db=st.session_state.vector_db,
                question=question,
                k=k
            )

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        if result["sources"]:
            for s in result["sources"]:
                st.markdown(f"- **{s['source']}** (chunk: {s.get('chunk_id', 'N/A')})")
        else:
            st.write("No sources found.")

        with st.expander("Retrieved Context"):
            st.write(result["context"])
