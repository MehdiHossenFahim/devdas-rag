"""Split chapters into overlapping chunks and attach citation metadata."""
from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Split on the largest natural boundary first, and only fall back to smaller
# ones when a piece is still too long:
#   paragraph -> line -> sentence end (Bengali danda "।", "?", "!") -> word.
# keep_separator="end" keeps the "।" attached to the sentence it finishes.
SEPARATORS = ["\n\n", "\n", "।", "?", "!", " ", ""]


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",
        length_function=len,
        add_start_index=True,
    )


def split_chapters(
    chapters: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Return LangChain Documents whose metadata is used later for citations."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    documents: list[Document] = []

    for ch in chapters:
        # add_start_index only works through create_documents
        docs = [d for d in splitter.create_documents([ch["text"]]) if d.page_content.strip()]
        total = len(docs)
        for i, d in enumerate(docs, 1):
            d.metadata = {
                "book": ch["book"],
                "chapter": ch["chapter"],
                "chapter_no": ch.get("chapter_no"),
                "section": f"{ch['chapter']} — অংশ {i}/{total}",
                "source": ch["url"],
                "chunk_index": i,
                "chunks_in_chapter": total,
                "start_index": d.metadata.get("start_index", 0),
                "chunk_id": f"ch{ch.get('chapter_no') or 0:02d}-{i:03d}",
            }
            d.page_content = d.page_content.strip()
            documents.append(d)
    return documents


def save_chunks(docs: list[Document]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d.page_content, **d.metadata}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from crawler import load_chapters

    docs = split_chapters(load_chapters())
    save_chunks(docs)
    sizes = [len(d.page_content) for d in docs]
    print(f"{len(docs)} chunks | avg {sum(sizes)/len(sizes):.0f} chars | max {max(sizes)}")
