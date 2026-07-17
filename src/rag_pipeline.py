from typing import Dict, Any, List
import numpy as np
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
        start = max(0, end - chunk_overlap)
    return chunks


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm


def build_vectorstore(documents, chunk_size: int = 800, chunk_overlap: int = 150):
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
    embeddings = _normalize(embeddings)

    return {
        "embedding_model": embedding_model,
        "texts": chunk_texts,
        "metas": chunk_metas,
        "embeddings": embeddings
    }


def answer_question(vector_db, question: str, k: int = 4) -> Dict[str, Any]:
    if vector_db is None:
        return {"answer": "Vector store is empty.", "context": "", "sources": []}

    embedding_model = vector_db["embedding_model"]
    texts = vector_db["texts"]
    metas = vector_db["metas"]
    embeddings = vector_db["embeddings"]

    q_emb = embedding_model.encode([question], convert_to_numpy=True).astype("float32")
    q_emb = _normalize(q_emb)

    sims = np.dot(embeddings, q_emb[0])  # cosine similarity
    k = min(k, len(texts))
    top_idx = np.argsort(-sims)[:k]

    retrieved_chunks = [texts[i] for i in top_idx]
    sources = [metas[i] for i in top_idx]

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
