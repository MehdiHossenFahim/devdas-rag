"""Full ingestion pipeline:

    Wikisource -> crawl -> clean -> chunk -> embed -> FAISS index

Usage
-----
    python ingest.py                 # crawl (cached) + build index
    python ingest.py --skip-crawl    # reuse data/chapters.jsonl
    python ingest.py --refresh       # ignore the crawl cache and re-download
    python ingest.py --chunk-size 1000 --chunk-overlap 200
"""
from __future__ import annotations

import argparse
import logging
import time

import config
from chunker import save_chunks, split_chapters
from crawler import crawl_book, load_chapters
from embeddings import get_embeddings
from vectorstore import build_index

log = logging.getLogger("ingest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-crawl", action="store_true", help="reuse data/chapters.jsonl")
    ap.add_argument("--refresh", action="store_true", help="re-download pages even if cached")
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1-2. Crawl + clean
    t0 = time.time()
    chapters = load_chapters() if args.skip_crawl else crawl_book(refresh=args.refresh)
    log.info("STEP 1-2  crawled + cleaned: %d chapters, %d characters",
             len(chapters), sum(len(c["text"]) for c in chapters))

    # 3. Chunk
    docs = split_chapters(chapters, args.chunk_size, args.chunk_overlap)
    save_chunks(docs)
    log.info("STEP 3    chunked: %d chunks (size=%d, overlap=%d)",
             len(docs), args.chunk_size, args.chunk_overlap)

    # 4-5. Embed + index
    embeddings = get_embeddings()
    log.info("STEP 4-5  embedding %d chunks with %s ... (this can take a few minutes on CPU)",
             len(docs), config.EMBEDDING_MODEL)
    build_index(
        docs, embeddings, config.VECTORSTORE_DIR,
        info={
            "book": config.BOOK_TITLE,
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "num_chapters": len(chapters),
            "chapters": [c["chapter"] for c in chapters],
        },
    )
    log.info("DONE      FAISS index saved to %s in %.0fs", config.VECTORSTORE_DIR, time.time() - t0)
    print("\nNext step:  streamlit run app.py")


if __name__ == "__main__":
    main()
