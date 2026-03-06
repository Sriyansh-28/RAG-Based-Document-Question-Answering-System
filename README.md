# RAG-Based-Document-Question-Answering-System

**NLP • LLMs • Information Retrieval**

This project demonstrates a Retrieval-Augmented Generation (RAG) workflow for answering user questions from custom document collections. The implementation style is designed to feel like working in **Google Colab inside VS Code** (notebook-driven, step-by-step cells).

## Project Highlights

- Built an end-to-end **RAG pipeline** to answer natural language questions grounded in document content.
- Implemented **document chunking** and **vector embeddings** for semantic retrieval.
- Integrated an **LLM response generation layer** that uses retrieved context to produce accurate, source-grounded answers.

---

## Colab-in-VS Code Development Style

If you use the Colab/Jupyter workflow in VS Code, structure development as sequential notebook cells:

1. **Environment setup cell**
2. **Document loading + preprocessing cell**
3. **Chunking + embedding cell**
4. **Vector index creation cell**
5. **Retriever + LLM chain cell**
6. **Inference/testing cell**

You can run this with a `.ipynb` file in VS Code (Jupyter extension) or with Python scripts using `# %%` cells.

---

## Notebook-Style Pipeline (VS Code Friendly)

```python
# %% [markdown]
# 1) Install dependencies (first run)

# %%
# !pip install -q langchain langchain-community sentence-transformers faiss-cpu pypdf

# %% [markdown]
# 2) Imports

# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# %% [markdown]
# 3) Load documents

# %%
loader = PyPDFLoader("docs/sample.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# %% [markdown]
# 4) Chunk documents

# %%
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# %% [markdown]
# 5) Create embeddings + vector store

# %%
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# %% [markdown]
# 6) Retrieve context for a query

# %%
query = "What is the key contribution of this document?"
retrieved_docs = retriever.get_relevant_documents(query)
context = "\n\n".join([d.page_content for d in retrieved_docs])
print(context[:1000])
```

---

## LLM Integration for Grounded Answers

In practice, pass the retrieved context to your selected LLM (OpenAI, Gemini, local model, etc.) with a grounding prompt:

```text
You are a helpful assistant. Answer only from the provided context.
If the answer is not in the context, say you do not know.

Context:
{retrieved_context}

Question:
{user_question}
```

This reduces hallucinations and keeps responses traceable to source chunks.

---

## Suggested Repository Layout

```text
RAG-Based-Document-Question-Answering-System/
├── notebooks/
│   └── rag_pipeline.ipynb
├── docs/
│   └── sample.pdf
├── src/
│   ├── ingest.py
│   ├── retriever.py
│   └── qa_chain.py
├── requirements.txt
└── README.md
```

---

## Evaluation Ideas

- **Retrieval quality:** Recall@k, MRR, or manual relevance checks.
- **Answer quality:** factuality, completeness, and groundedness.
- **Latency:** chunking/indexing time and query response time.

---

## Resume-Ready Project Description

Developed a Retrieval-Augmented Generation (RAG) pipeline for document question answering using NLP, vector search, and LLM-based response generation. Implemented document chunking and semantic embeddings to retrieve relevant context, then generated accurate, context-grounded answers through prompt-engineered LLM integration. Built and iterated in a Colab-style notebook workflow inside VS Code for fast experimentation and reproducible development.
