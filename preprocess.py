"""Turn Wikisource HTML into clean plain text.

Steps
-----
1. `html_to_text`  – keep only the readable prose (drop headers, page numbers,
                     edit links, footnote markers, tables, navigation, ...).
2. `clean_text`    – Unicode NFC normalisation, remove invisible characters,
                     collapse whitespace, drop page-number / header residue.
"""
from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dd", "dt", "blockquote"]

# CSS selectors for things that are NOT part of the book text.
NOISE_SELECTORS = [
    "style", "script", "table", "sup.reference", ".reference", "ol.references",
    ".mw-editsection", ".noprint", ".ws-noexport", "#headerContainer",
    ".ws-header", ".pagenum", ".ws-pagenum", ".mw-empty-elt", ".toc", "#toc",
    ".printfooter", ".catlinks", ".prp-page-qualityheader", ".navbox",
    ".ambox",
]

BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
_DIGIT = rf"[0-9{BENGALI_DIGITS}]"

# invisible characters that only add noise (ZWJ/ZWNJ are *kept*: they matter
# for Bengali conjunct rendering, e.g. "র‍্যা").
_INVISIBLE = dict.fromkeys(
    map(ord, ["\u200b", "\u200e", "\u200f", "\u2060", "\ufeff", "\u00ad", "\u202a", "\u202c"]),
    None,
)
_PAGE_RANGE_RE = re.compile(rf"^{_DIGIT}+\s*[-–—]\s*{_DIGIT}+$")
_ONLY_NUMBER_RE = re.compile(rf"^{_DIGIT}+$")


def html_to_text(html: str) -> str:
    """Extract paragraph text from a rendered MediaWiki page."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup

    for selector in NOISE_SELECTORS:
        for el in root.select(selector):
            el.decompose()

    for br in root.find_all("br"):
        br.replace_with("\n")

    blocks: list[str] = []
    for el in root.find_all(BLOCK_TAGS):
        if el.find_parent(BLOCK_TAGS):  # avoid duplicating nested blocks
            continue
        text = el.get_text("")
        if text.strip():
            blocks.append(text.strip())

    if not blocks:  # unusual page layout – fall back to everything
        return root.get_text("\n")
    return "\n\n".join(blocks)


def clean_text(
    text: str,
    *,
    drop_lines: set[str] | None = None,
    header_markers: list[str] | None = None,
) -> str:
    """Normalise and tidy extracted text.

    drop_lines      exact lines to remove (page furniture such as the book title
                    or the chapter title repeated as a header).
    header_markers  strings such as [book title, author, chapter name]. A *short*
                    line containing at least two of them is treated as the
                    Wikisource header template (e.g. "দেবদাসশরৎচন্দ্র
                    চট্টোপাধ্যায়ষোড়শ পরিচ্ছেদ১০০-১১০") and dropped.
    """
    nfc = lambda s: unicodedata.normalize("NFC", s).strip()  # noqa: E731
    drop_lines = {nfc(d) for d in (drop_lines or set())}
    header_markers = [nfc(m) for m in (header_markers or []) if m]

    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = text.replace("\u00a0", " ").replace("\r", "")

    cleaned_lines: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if line in drop_lines:
            continue
        if len(line) <= 160 and sum(m in line for m in header_markers) >= 2:
            continue
        if _PAGE_RANGE_RE.match(line) or _ONLY_NUMBER_RE.match(line):
            continue  # "১০০-১১০" or a bare page number
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_query(text: str) -> str:
    """Apply the same Unicode normalisation to user questions."""
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    return re.sub(r"\s+", " ", text).strip()
