"""Multilingual (Bengali-capable) embedding models."""
from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

import config

log = logging.getLogger(__name__)


class PrefixedEmbeddings(Embeddings):
    """Adds the instruction prefixes some models were trained with.

    multilingual-e5 models expect  "query: <question>"  and  "passage: <text>".
    Forgetting the prefixes noticeably hurts retrieval quality.
    """

    def __init__(self, base: Embeddings, query_prefix: str = "", doc_prefix: str = ""):
        self.base = base
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.query_prefix + text)


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_embeddings(model_name: str | None = None, device: str | None = None) -> Embeddings:
    """Load a sentence-transformers model through LangChain."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = model_name or config.EMBEDDING_MODEL
    device = _pick_device(device or config.EMBEDDING_DEVICE)
    log.info("Loading embedding model %s on %s (first run downloads it)", model_name, device)

    base = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        show_progress=True,
        # normalised vectors -> L2 distance ranking == cosine-similarity ranking
        encode_kwargs={"normalize_embeddings": True, "batch_size": config.EMBEDDING_BATCH_SIZE},
    )
    # Bengali chunks of ~800 chars are far below 512 tokens; capping the sequence
    # length keeps CPU encoding fast (bge-m3 defaults to 8192).
    try:
        base._client.max_seq_length = config.EMBEDDING_MAX_SEQ_LEN
    except Exception:  # pragma: no cover - attribute differs between versions
        log.debug("Could not set max_seq_length; using model default")

    if "e5" in model_name.lower():
        return PrefixedEmbeddings(base, query_prefix="query: ", doc_prefix="passage: ")
    return base
