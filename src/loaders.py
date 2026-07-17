import os
import tempfile
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)


def _save_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def load_uploaded_files(uploaded_files) -> List[Document]:
    all_docs: List[Document] = []

    for uf in uploaded_files:
        ext = os.path.splitext(uf.name.lower())[1]
        temp_path = _save_to_temp(uf)

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(temp_path)
                docs = loader.load()

            elif ext == ".docx":
                loader = Docx2txtLoader(temp_path)
                docs = loader.load()

            elif ext in [".txt", ".md"]:
                loader = TextLoader(temp_path, encoding="utf-8")
                docs = loader.load()

            elif ext == ".csv":
                loader = CSVLoader(file_path=temp_path, encoding="utf-8")
                docs = loader.load()

            else:
                docs = []

            for i, d in enumerate(docs):
                d.metadata["source"] = uf.name
                d.metadata["chunk_id"] = i
            all_docs.extend(docs)

        except Exception:
            continue
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return all_docs
