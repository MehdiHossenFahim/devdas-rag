"""Central configuration for the Devdas RAG chatbot.

Every value can be overridden with an environment variable or a `.env` file
(see `.env.example`).
"""
from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # cached raw API responses (HTML)
CHAPTERS_FILE = DATA_DIR / "chapters.jsonl"  # cleaned chapter texts
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"      # final chunks (for inspection)
VECTORSTORE_DIR = ROOT / "vectorstore"       # FAISS index lives here
RESULTS_DIR = ROOT / "results"
TEST_QUESTIONS_FILE = ROOT / "test_questions.json"

# --------------------------------------------------------------------------- #
# Book
# --------------------------------------------------------------------------- #
BOOK_URL = (
    "https://bn.wikisource.org/wiki/"
    "%E0%A6%A6%E0%A7%87%E0%A6%AC%E0%A6%A6%E0%A6%BE%E0%A6%B8_"
    "(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_"
    "%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D"
    "%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC)"
)
BOOK_TITLE = "দেবদাস"
BOOK_AUTHOR = "শরৎচন্দ্র চট্টোপাধ্যায়"

WIKI_HOST = "https://bn.wikisource.org"
API_URL = f"{WIKI_HOST}/w/api.php"

# Wikisource page title, e.g. "দেবদাস (শরৎচন্দ্র চট্টোপাধ্যায়)"
PAGE_TITLE = unicodedata.normalize(
    "NFC", unquote(BOOK_URL.split("/wiki/", 1)[1])
).replace("_", " ")

# --------------------------------------------------------------------------- #
# Crawler (polite by default: Wikisource rate-limits aggressive clients)
# --------------------------------------------------------------------------- #
# Wikimedia asks every client to send a descriptive User-Agent with contact info.
USER_AGENT = _env(
    "USER_AGENT",
    "DevdasRAGStudentProject/1.0 (educational RAG assignment; "
    "contact: your-email@example.com)",
)
REQUEST_DELAY = float(_env("REQUEST_DELAY", "2.0"))   # min seconds between requests
MAX_RETRIES = int(_env("MAX_RETRIES", "6"))           # per request

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE = int(_env("CHUNK_SIZE", "800"))       # characters
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "150"))  # characters

# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = _env("EMBEDDING_DEVICE", "auto")  # auto | cpu | cuda | mps
EMBEDDING_BATCH_SIZE = int(_env("EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_MAX_SEQ_LEN = int(_env("EMBEDDING_MAX_SEQ_LEN", "512"))

# --------------------------------------------------------------------------- #
# Retrieval + LLM
# --------------------------------------------------------------------------- #
TOP_K = int(_env("TOP_K", "5"))

GROQ_API_KEY = _env("GROQ_API_KEY", "")
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-120b")
# gpt-oss models accept: low | medium | high. "low" keeps token usage small,
# which matters on Groq's free tier (tokens-per-minute limit).
GROQ_REASONING_EFFORT = _env("GROQ_REASONING_EFFORT", "low")
# gpt-oss "reasoning" tokens count towards max_tokens, so leave headroom.
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "2048"))

# Sentinel the LLM must output when the context does not contain the answer.
NOT_IN_BOOK_TOKEN = "NOT_IN_BOOK"
