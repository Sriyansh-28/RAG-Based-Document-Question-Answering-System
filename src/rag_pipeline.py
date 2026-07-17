from typing import Dict, Any, List
import re


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


def tokenize(s: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", s.lower()))


def build_vectorstore(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    chunk_texts = []
    chunk_metas = []
    chunk_tokens = []

    for doc in documents:
        text = doc.get("page_content", "")
        meta = doc.get("metadata", {})
        source = meta.get("source", "unknown")

        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, ch in enumerate(pieces):
            ch = ch.strip()
            if ch:
                chunk_texts.append(ch)
                chunk_metas.append({"source": source, "chunk_id": i})
                chunk_tokens.append(tokenize(ch))

    if not chunk_texts:
        return None

    return {
        "texts": chunk_texts,
        "metas": chunk_metas,
        "tokens": chunk_tokens
    }


def answer_question(vector_db, question: str, k: int = 4) -> Dict[str, Any]:
    if vector_db is None:
        return {"answer": "Vector store is empty.", "context": "", "sources": []}

    texts = vector_db["texts"]
    metas = vector_db["metas"]
    tokens = vector_db["tokens"]

    q_tokens = tokenize(question)
    if not q_tokens:
        return {"answer": "Please ask a valid question.", "context": "", "sources": []}

    # Jaccard similarity
    scores = []
    for i, tks in enumerate(tokens):
        inter = len(q_tokens & tks)
        union = len(q_tokens | tks) or 1
        score = inter / union
        scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = [i for s, i in scores[: max(1, min(k, len(scores)))] if s > 0]

    if not top:
        return {
            "answer": "I do not know based on the provided documents.",
            "context": "",
            "sources": []
        }

    retrieved_chunks = [texts[i] for i in top]
    sources = [metas[i] for i in top]
    context = "\n\n".join(retrieved_chunks)[:5000]

    # Lightweight grounded answer: return best matching chunk snippet
    best_chunk = retrieved_chunks[0]
    answer = best_chunk[:700]
    if len(best_chunk) > 700:
        answer += "..."

    return {"answer": answer, "context": context, "sources": sources}
