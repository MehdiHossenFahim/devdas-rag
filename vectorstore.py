"""FAISS vector store helpers (build, save, load, sanity-check)."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config

INFO_FILE = "index_info.json"


def build_index(
    docs: list[Document],
    embeddings: Embeddings,
    path: Path | None = None,
    info: dict | None = None,
) -> FAISS:
    """Embed all chunks and build a FAISS index. Saves it when `path` is given."""
    store = FAISS.from_documents(docs, embeddings)
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))
        (path / INFO_FILE).write_text(
            json.dumps({"num_chunks": len(docs), **(info or {})}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return store


def index_exists(path: Path | None = None) -> bool:
    path = path or config.VECTORSTORE_DIR
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def read_info(path: Path | None = None) -> dict:
    path = path or config.VECTORSTORE_DIR
    f = path / INFO_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def load_index(embeddings: Embeddings, path: Path | None = None) -> FAISS:
    path = path or config.VECTORSTORE_DIR
    if not index_exists(path):
        raise FileNotFoundError(
            f"No FAISS index found in {path}. Build it first with:  python ingest.py"
        )
    info = read_info(path)
    if info.get("embedding_model") and info["embedding_model"] != config.EMBEDDING_MODEL:
        raise RuntimeError(
            f"The index was built with '{info['embedding_model']}' but EMBEDDING_MODEL is "
            f"'{config.EMBEDDING_MODEL}'. Set them to the same value or re-run `python ingest.py`."
        )
    # The pickle is created by our own ingest step, so deserialising it is safe.
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
