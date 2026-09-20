"""Evaluation utilities for the 10 test questions + the bonus comparison.

    python evaluate.py --locate [--write]   find which chapters contain the evidence
                                            keywords (fills `expected_source`, rewrites
                                            TEST_QUESTIONS.md)
    python evaluate.py --retrieval          hit-rate@k of the current FAISS index
    python evaluate.py --qa                 run all questions through the full RAG chain
                                            (uses the Groq API) -> results/qa_results.md
    python evaluate.py --compare-chunking 400:80 800:150 1200:200
    python evaluate.py --compare-embeddings BAAI/bge-m3 intfloat/multilingual-e5-base

Hit definition (retrieval only, no LLM): a question is a *hit@k* if any of the top-k
retrieved chunks
  (a) belongs to one of the question's `expected_source` chapters (if any are set) AND
  (b) contains at least one of the question's `evidence_keywords`.
Questions with `answerable: false` are excluded from the hit-rate.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import unicodedata
from collections import Counter

import config
from crawler import load_chapters

KS = (1, 3, 5)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_questions() -> list[dict]:
    qs = json.loads(config.TEST_QUESTIONS_FILE.read_text(encoding="utf-8"))
    for q in qs:
        q["evidence_keywords"] = [nfc(k) for k in q.get("evidence_keywords", [])]
        q["expected_source"] = [nfc(s) for s in q.get("expected_source", [])]
    return qs


def save_questions(qs: list[dict]) -> None:
    config.TEST_QUESTIONS_FILE.write_text(
        json.dumps(qs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


# --------------------------------------------------------------------------- #
# TEST_QUESTIONS.md
# --------------------------------------------------------------------------- #
def write_test_table(qs: list[dict]) -> None:
    lines = [
        "# Test Questions",
        "",
        "10 test questions (8 answerable + 2 that are **not** in the book, to test no-answer handling).",
        "",
        "| # | Question | Expected Answer | Source / Chapter |",
        "|---|----------|-----------------|------------------|",
    ]
    for q in qs:
        src = ", ".join(q["expected_source"]) if q["expected_source"] else (
            "— (not in the book)" if not q["answerable"] else "⚠️ run `python evaluate.py --locate --write`"
        )
        lines.append(
            f"| {q['id']} | {md_escape(q['question'])} | {md_escape(q['expected_answer'])} | {md_escape(src)} |"
        )
    (config.ROOT / "TEST_QUESTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote TEST_QUESTIONS.md")


# --------------------------------------------------------------------------- #
# --locate
# --------------------------------------------------------------------------- #
def locate(write: bool) -> None:
    chapters = load_chapters()
    qs = load_questions()
    for q in qs:
        if not q["answerable"]:
            continue
        counts = {
            ch["chapter"]: {k: nfc(ch["text"]).count(k) for k in q["evidence_keywords"]}
            for ch in chapters
        }
        matching = {c: v for c, v in counts.items() if any(v.values())}
        print(f"\nQ{q['id']}: {q['question']}")
        if not matching:
            print("   ⚠️  none of the evidence keywords occur in the book text – edit "
                  "`evidence_keywords` in test_questions.json (check the spelling used in the book)")
            continue
        for c, v in sorted(matching.items(), key=lambda kv: -sum(kv[1].values())):
            print(f"   {c:<28} {dict((k, n) for k, n in v.items() if n)}")
        # keyword-based suggestion: the two chapters with the most keyword occurrences
        ranked = sorted(matching, key=lambda c: -sum(matching[c].values()))[:2]
        if write and not q["expected_source"]:
            q["expected_source"] = ranked
            print(f"   -> expected_source = {ranked}")
    if write:
        save_questions(qs)
        write_test_table(qs)
        print("\nReview the chapters written to test_questions.json – they are only a "
              "keyword-based suggestion. Edit them by hand where needed, then re-run with "
              "--write-table.")


# --------------------------------------------------------------------------- #
# Retrieval hit-rate
# --------------------------------------------------------------------------- #
def is_hit(doc, q: dict) -> bool:
    exp = q["expected_source"]
    if exp and doc.metadata["chapter"] not in exp:
        return False
    kws = q["evidence_keywords"]
    if kws:
        return any(k in doc.page_content for k in kws)
    return bool(exp)


def hit_rates(vectorstore, qs: list[dict]) -> tuple[dict[int, float], list[dict]]:
    answerable = [q for q in qs if q["answerable"]]
    hits = Counter()
    details = []
    for q in answerable:
        docs = vectorstore.similarity_search(nfc(q["question"]), k=max(KS))
        rank = next((i for i, d in enumerate(docs, 1) if is_hit(d, q)), None)
        for k in KS:
            hits[k] += int(rank is not None and rank <= k)
        details.append({"id": q["id"], "question": q["question"], "rank": rank,
                        "top_chapters": [d.metadata["chapter"] for d in docs]})
    return {k: hits[k] / len(answerable) for k in KS}, details


def run_retrieval() -> None:
    from embeddings import get_embeddings
    from vectorstore import load_index

    qs = load_questions()
    rates, details = hit_rates(load_index(get_embeddings()), qs)
    print(f"\nModel: {config.EMBEDDING_MODEL}  chunk={config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}")
    for k in KS:
        print(f"  Hit@{k}: {rates[k]:.0%}")
    for d in details:
        print(f"  Q{d['id']:<2} first hit rank: {d['rank']}   {d['question']}")


# --------------------------------------------------------------------------- #
# Bonus: compare two (or more) approaches
# --------------------------------------------------------------------------- #
def _table(rows: list[tuple[str, int, dict[int, float]]], title: str) -> str:
    out = [f"### {title}", "", "| Approach | # chunks | Hit@1 | Hit@3 | Hit@5 |", "|---|---|---|---|---|"]
    for name, n, r in rows:
        out.append(f"| {name} | {n} | {r[1]:.0%} | {r[3]:.0%} | {r[5]:.0%} |")
    return "\n".join(out)


def compare_chunking(configs: list[str]) -> None:
    from langchain_community.vectorstores import FAISS

    from chunker import split_chapters
    from embeddings import get_embeddings

    chapters, qs, emb = load_chapters(), load_questions(), get_embeddings()
    rows = []
    for spec in configs:
        size, overlap = (int(x) for x in spec.split(":"))
        docs = split_chapters(chapters, size, overlap)
        print(f"Embedding {len(docs)} chunks for chunk_size={size}, overlap={overlap} ...")
        rates, _ = hit_rates(FAISS.from_documents(docs, emb), qs)
        rows.append((f"chunk_size={size}, overlap={overlap}", len(docs), rates))
    _save_and_print(rows, f"Chunking comparison (embedding model: {config.EMBEDDING_MODEL})",
                    "bonus_chunking.md")


def compare_embeddings(models: list[str]) -> None:
    from langchain_community.vectorstores import FAISS

    from chunker import split_chapters
    from embeddings import get_embeddings

    chapters, qs = load_chapters(), load_questions()
    docs = split_chapters(chapters, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    rows = []
    for model in models:
        print(f"Embedding {len(docs)} chunks with {model} ...")
        rates, _ = hit_rates(FAISS.from_documents(docs, get_embeddings(model)), qs)
        rows.append((model, len(docs), rates))
    _save_and_print(rows, f"Embedding-model comparison (chunk {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP})",
                    "bonus_embeddings.md")


def _save_and_print(rows, title: str, filename: str) -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    n_q = sum(q["answerable"] for q in load_questions())
    text = _table(rows, title) + f"\n\n_Hit@k over {n_q} answerable test questions._\n"
    (config.RESULTS_DIR / filename).write_text(text, encoding="utf-8")
    print("\n" + text)


# --------------------------------------------------------------------------- #
# Full RAG run (LLM)
# --------------------------------------------------------------------------- #
def run_qa(delay: float) -> None:
    from rag import BookRAG

    qs, bot = load_questions(), BookRAG()
    lines = ["# QA results", "", f"LLM: `{config.GROQ_MODEL}` · Embeddings: `{config.EMBEDDING_MODEL}` "
             f"· chunk {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} · top-k {config.TOP_K}", "",
             "| # | Question | Chatbot answer | Cited sections | Expected | Behaviour OK? |",
             "|---|---|---|---|---|---|"]
    ok_count = 0
    for i, q in enumerate(qs):
        if i:
            time.sleep(delay)  # stay under Groq's free-tier tokens/minute limit
        res = bot.ask(q["question"])
        behaviour_ok = res.found == q["answerable"]
        if q["expected_source"] and res.found:
            behaviour_ok = behaviour_ok and any(s.chapter in q["expected_source"] for s in res.sources)
        ok_count += behaviour_ok
        print(f"Q{q['id']}: found={res.found} ok={behaviour_ok}\n   {res.answer[:150]}")
        lines.append(
            f"| {q['id']} | {md_escape(q['question'])} | {md_escape(res.answer)} | "
            f"{md_escape(res.citation_text or '—')} | {md_escape(q['expected_answer'])} | "
            f"{'✅' if behaviour_ok else '❌'} |"
        )
    lines += ["", f"**{ok_count}/{len(qs)}** answered/declined as expected "
              "(automatic check; read the answers yourself to judge quality)."]
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "qa_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{ok_count}/{len(qs)} OK -> results/qa_results.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locate", action="store_true")
    ap.add_argument("--write", action="store_true", help="with --locate: save suggested chapters")
    ap.add_argument("--write-table", action="store_true", help="only regenerate TEST_QUESTIONS.md")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--qa-delay", type=float, default=20.0, help="seconds between LLM calls")
    ap.add_argument("--compare-chunking", nargs="+", metavar="SIZE:OVERLAP")
    ap.add_argument("--compare-embeddings", nargs="+", metavar="MODEL")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.write_table:
        write_test_table(load_questions())
    if args.locate:
        locate(args.write)
    if args.retrieval:
        run_retrieval()
    if args.compare_chunking:
        compare_chunking(args.compare_chunking)
    if args.compare_embeddings:
        compare_embeddings(args.compare_embeddings)
    if args.qa:
        run_qa(args.qa_delay)
    if not (args.write_table or args.locate or args.retrieval or args.qa
            or args.compare_chunking or args.compare_embeddings):
        ap.print_help()


if __name__ == "__main__":
    main()
