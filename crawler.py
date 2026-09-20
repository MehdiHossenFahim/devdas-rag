"""Rate-limit-friendly crawler for a Bengali Wikisource book.

Why the MediaWiki API instead of scraping /wiki/ pages?
  * it is the officially supported (and rate-limit friendly) way to read pages;
  * `action=parse` returns the fully rendered chapter, including text that is
    transcluded from the proofread "পাতা:" (Page:) namespace;
  * `list=allpages` lets us discover *every* subpage of the book, so nothing
    is missed even if the main page's table of contents is incomplete.

Politeness / rate-limit handling
  * strictly sequential requests (no threads);
  * minimum delay (+ random jitter) between requests  -> REQUEST_DELAY;
  * descriptive User-Agent, as Wikimedia requires;
  * `maxlag` parameter so we back off when the servers are busy;
  * HTTP 429/5xx  -> honour `Retry-After`, else exponential back-off;
  * every fetched page is cached in data/raw/, so re-running never re-downloads
    (delete the folder or pass --refresh to force a fresh crawl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

import config
from preprocess import clean_text, html_to_text

log = logging.getLogger("crawler")

# Bengali ordinal words -> chapter number (used for sorting / metadata).
_ORDINALS = {
    "প্রথম": 1, "দ্বিতীয়": 2, "তৃতীয়": 3, "চতুর্থ": 4, "পঞ্চম": 5, "ষষ্ঠ": 6,
    "সপ্তম": 7, "অষ্টম": 8, "নবম": 9, "দশম": 10, "একাদশ": 11, "দ্বাদশ": 12,
    "ত্রয়োদশ": 13, "চতুর্দ্দশ": 14, "চতুর্দশ": 14, "পঞ্চদশ": 15, "ষোড়শ": 16,
    "সপ্তদশ": 17, "অষ্টাদশ": 18, "ঊনবিংশ": 19, "বিংশ": 20,
}
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def chapter_number(chapter_name: str) -> int | None:
    """'ষোড়শ পরিচ্ছেদ' -> 16 ; unknown names -> None."""
    name = unicodedata.normalize("NFC", chapter_name).strip()
    first = name.split()[0] if name.split() else ""
    if first in _ORDINALS:
        return _ORDINALS[first]
    digits = re.search(r"[0-9০-৯]+", name)
    if digits:
        return int(digits.group().translate(_BN_DIGITS))
    return None


def wiki_url(title: str) -> str:
    return f"{config.WIKI_HOST}/wiki/{quote(title.replace(' ', '_'), safe='/:()')}"


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
class PoliteClient:
    def __init__(self, min_delay: float, max_retries: int, user_agent: str):
        self.min_delay = min_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip"})
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = self.min_delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait + random.uniform(0.0, 0.5))

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(90.0, 5.0 * 2 ** (attempt - 1)) + random.uniform(0, 1.5)

    def get_json(self, params: dict) -> dict:
        params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(config.API_URL, params=params, timeout=45)
            except requests.RequestException as exc:
                self._last_request = time.monotonic()
                wait = self._backoff(attempt)
                log.warning("Network error (%s). Retry %d/%d in %.0fs",
                            exc.__class__.__name__, attempt, self.max_retries, wait)
                time.sleep(wait)
                continue
            self._last_request = time.monotonic()

            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) + 1 if retry_after.isdigit() else self._backoff(attempt)
                log.warning("HTTP %d (rate limited / busy). Retry %d/%d in %.0fs",
                            resp.status_code, attempt, self.max_retries, wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            error = data.get("error")
            if error:
                if error.get("code") in ("maxlag", "ratelimited"):
                    wait = self._backoff(attempt)
                    log.warning("API says '%s'. Retry %d/%d in %.0fs",
                                error["code"], attempt, self.max_retries, wait)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"MediaWiki API error: {error}")
            return data
        raise RuntimeError(f"Giving up after {self.max_retries} attempts: {params}")


# --------------------------------------------------------------------------- #
# Discovery of chapter pages
# --------------------------------------------------------------------------- #
def _norm_title(title: str) -> str:
    return unicodedata.normalize("NFC", title).replace("_", " ").strip()


def _toc_titles(client: PoliteClient, cache_dir: Path, refresh: bool) -> list[str]:
    """Subpage titles linked from the main page, in table-of-contents order."""
    main = fetch_page(client, config.PAGE_TITLE, cache_dir, refresh)
    soup = BeautifulSoup(main["html"], "html.parser")
    prefix = config.PAGE_TITLE + "/"
    titles: list[str] = []
    for a in soup.select("a[href]"):
        if "new" in (a.get("class") or []):  # red link = page does not exist
            continue
        title = _norm_title(a.get("title") or "")
        if title.startswith(prefix) and title not in titles:
            titles.append(title)
    return titles


def _allpages_titles(client: PoliteClient) -> list[str]:
    """Every page whose title starts with '<book>/' (handles API pagination)."""
    titles: list[str] = []
    params: dict = {
        "action": "query", "list": "allpages", "apnamespace": 0,
        "apprefix": config.PAGE_TITLE + "/", "aplimit": "max",
    }
    while True:
        data = client.get_json(params)
        titles += [_norm_title(p["title"]) for p in data["query"]["allpages"]]
        if "continue" not in data:
            return titles
        params = {**params, **data["continue"]}


def discover_chapters(client: PoliteClient, cache_dir: Path, refresh: bool) -> list[str]:
    toc = _toc_titles(client, cache_dir, refresh)
    extra = [t for t in _allpages_titles(client) if t not in toc]
    extra.sort(key=lambda t: (chapter_number(t.rsplit("/", 1)[-1]) or 10**6, t))
    titles = toc + extra
    log.info("Discovered %d subpages (%d from main-page TOC, %d extra via allpages)",
             len(titles), len(toc), len(extra))
    return titles


# --------------------------------------------------------------------------- #
# Fetching (with on-disk cache)
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: Path, title: str) -> Path:
    return cache_dir / f"{hashlib.sha1(title.encode('utf-8')).hexdigest()[:12]}.json"


def fetch_page(client: PoliteClient, title: str, cache_dir: Path, refresh: bool = False) -> dict:
    path = _cache_path(cache_dir, title)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    data = client.get_json({
        "action": "parse", "page": title, "prop": "text", "redirects": 1,
        "disableeditsection": 1, "disablelimitreport": 1, "disabletoc": 1,
    })
    page = {
        "title": _norm_title(data["parse"]["title"]),
        "requested_title": title,
        "html": data["parse"]["text"],
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
    return page


def crawl_book(refresh: bool = False) -> list[dict]:
    """Crawl every chapter, clean it and return a list of chapter records."""
    client = PoliteClient(config.REQUEST_DELAY, config.MAX_RETRIES, config.USER_AGENT)
    titles = discover_chapters(client, config.RAW_DIR, refresh)
    if not titles:
        raise RuntimeError(
            "No chapter subpages were found. Check BOOK_URL in config.py and your "
            "internet connection."
        )

    chapters: list[dict] = []
    for i, title in enumerate(titles, 1):
        chapter_name = title.rsplit("/", 1)[-1]
        cached = _cache_path(config.RAW_DIR, title).exists() and not refresh
        log.info("[%d/%d] %s%s", i, len(titles), chapter_name, "  (cached)" if cached else "")
        page = fetch_page(client, title, config.RAW_DIR, refresh)

        text = clean_text(
            html_to_text(page["html"]),
            drop_lines={config.BOOK_TITLE, config.BOOK_AUTHOR, chapter_name, page["title"]},
            header_markers=[config.BOOK_TITLE, config.BOOK_AUTHOR, chapter_name],
        )
        if len(text) < 30:
            log.warning("   -> almost no text extracted from '%s' (skipped)", chapter_name)
            continue
        chapters.append({
            "book": config.BOOK_TITLE,
            "author": config.BOOK_AUTHOR,
            "title": title,
            "chapter": chapter_name,
            "chapter_no": chapter_number(chapter_name),
            "url": wiki_url(title),
            "text": text,
        })
        log.info("   -> %d characters", len(text))

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with config.CHAPTERS_FILE.open("w", encoding="utf-8") as f:
        for ch in chapters:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    total = sum(len(c["text"]) for c in chapters)
    log.info("Saved %d chapters (%d characters) to %s", len(chapters), total, config.CHAPTERS_FILE)
    return chapters


def load_chapters() -> list[dict]:
    if not config.CHAPTERS_FILE.exists():
        raise FileNotFoundError(
            f"{config.CHAPTERS_FILE} not found. Run `python ingest.py` (or `python crawler.py`) first."
        )
    with config.CHAPTERS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Crawl the book from Bengali Wikisource")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    crawl_book(refresh=ap.parse_args().refresh)
