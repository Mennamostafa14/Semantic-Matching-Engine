# helpers/file_loader.py
import os
import tempfile

from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader


def load_from_bytes(file_bytes: bytes, file_name: str) -> list[Document]:
    """
    Write raw bytes to a temporary file and load it with the appropriate
    LangChain document loader.

    Supported formats: .pdf, .txt
    Unsupported formats return an empty list (no exception raised).

    The temp file is always deleted after loading, even on failure.
    """
    file_ext = os.path.splitext(file_name)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if file_ext == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif file_ext == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        else:
            return []
        return loader.load()
    finally:
        os.unlink(tmp_path)