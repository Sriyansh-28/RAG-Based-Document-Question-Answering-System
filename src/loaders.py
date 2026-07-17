import os
import tempfile
import pandas as pd
import docx

# Safe PDF import with fallback
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def _save_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def load_uploaded_files(uploaded_files):
    """
    Returns list of dict docs:
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
                    raise ImportError("PyPDF2 is not installed in environment.")
                reader = PdfReader(temp_path)
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages)

            elif ext == ".docx":
                d = docx.Document(temp_path)
                text = "\n".join([p.text for p in d.paragraphs])

            elif ext in [".txt", ".md"]:
                with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            elif ext == ".csv":
                df = pd.read_csv(temp_path)
                text = df.to_csv(index=False)

            else:
                # unsupported extension -> skip
                text = ""

            if text and text.strip():
                all_docs.append({
                    "page_content": text,
                    "metadata": {
                        "source": uf.name,
                        "chunk_id": 0
                    }
                })

        except Exception:
            # Continue with other files if one fails
            continue

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return all_docs
