import os
import tempfile
import pandas as pd
import streamlit as st

# Optional imports
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None


def _save_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    data = uploaded_file.getvalue()  # safer for Streamlit UploadedFile
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def load_uploaded_files(uploaded_files):
    """
    Returns list of docs in format:
    [
      {"page_content": "...", "metadata": {"source": "file.ext", "chunk_id": 0}},
      ...
    ]
    """
    all_docs = []

    for uf in uploaded_files:
        ext = os.path.splitext(uf.name.lower())[1]
        temp_path = _save_to_temp(uf)

        try:
            text = ""

            if ext == ".pdf":
                if PdfReader is None:
                    st.warning(f"PyPDF2 not available. Skipping: {uf.name}")
                    continue

                reader = PdfReader(temp_path)
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages).strip()

                # Key fix: handle scanned/image-only PDFs
                if not text:
                    st.error(
                        f"{uf.name}: No selectable text found (likely scanned/image PDF). "
                        "Please upload OCR-converted PDF or TXT/DOCX."
                    )
                    continue

            elif ext == ".docx":
                if docx is None:
                    st.warning(f"python-docx not available. Skipping: {uf.name}")
                    continue
                d = docx.Document(temp_path)
                text = "\n".join([p.text for p in d.paragraphs]).strip()

            elif ext in [".txt", ".md"]:
                with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()

            elif ext == ".csv":
                try:
                    df = pd.read_csv(temp_path)
                except Exception:
                    df = pd.read_csv(temp_path, encoding="latin-1")
                text = df.to_csv(index=False).strip()

            else:
                st.info(f"Unsupported file type skipped: {uf.name}")
                continue

            if text:
                all_docs.append({
                    "page_content": text,
                    "metadata": {"source": uf.name, "chunk_id": 0}
                })
                st.success(f"Loaded: {uf.name} ({len(text)} chars)")
            else:
                st.warning(f"No readable content found in: {uf.name}")

        except Exception as e:
            st.error(f"Failed to process {uf.name}: {e}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return all_docs
