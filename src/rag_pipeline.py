from typing import Dict, Any, List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - chunk_overlap
    return chunks


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")


def build_vectorstore(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    documents format expected from loaders.py:
    [
      {"page_content": "...", "metadata": {"source": "...", "chunk_id": 0}},
      ...
    ]
    """
    embedding_model = get_embedding_model()

    chunk_texts = []
    chunk_metas = []

    for doc in documents:
        text = doc.get("page_content", "")
        meta = doc.get("metadata", {})
        source = meta.get("source", "unknown")

        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, ch in enumerate(pieces):
            if ch.strip():
                chunk_texts.append(ch)
                chunk_metas.append({"source": source, "chunk_id": i})

    if not chunk_texts:
        return None

    embeddings = embedding_model.encode(chunk_texts, convert_to_numpy=True).astype("float32")

    # cosine similarity via normalized vectors + inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "index": index,
        "embedding_model": embedding_model,
        "texts": chunk_texts,
        "metas": chunk_metas
    }


def answer_question(vector_db, question: str, k: int = 4) -> Dict[str, Any]:
    if vector_db is None:
        return {"answer": "Vector store is empty.", "context": "", "sources": []}

    embedding_model = vector_db["embedding_model"]
    index = vector_db["index"]
    texts = vector_db["texts"]
    metas = vector_db["metas"]

    q_emb = embedding_model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    k = min(k, len(texts))
    scores, ids = index.search(q_emb, k)

    retrieved_chunks = []
    sources = []
    for idx in ids[0]:
        if 0 <= idx < len(texts):
            retrieved_chunks.append(texts[idx])
            sources.append(metas[idx])

    context = "\n\n".join(retrieved_chunks)[:5000]

    prompt = f"""You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not present in context, say "I do not know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

    generator = get_generator()
    answer = generator(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]

    return {"answer": answer, "context": context, "sources": sources}
