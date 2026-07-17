from typing import Dict, Any, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


def build_vectorstore(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


def answer_question(vector_db, question: str, k: int = 4) -> Dict[str, Any]:
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([d.page_content for d in retrieved_docs])[:5000]

    prompt = f"""You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not present in context, say "I do not know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    answer = generator(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]

    sources: List[Dict[str, Any]] = []
    for d in retrieved_docs:
        sources.append(
            {
                "source": d.metadata.get("source", "unknown"),
                "chunk_id": d.metadata.get("chunk_id", "N/A"),
            }
        )

    return {"answer": answer, "context": context, "sources": sources}
