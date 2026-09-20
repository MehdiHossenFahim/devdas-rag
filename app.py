"""Streamlit chat UI for the Devdas knowledge-base chatbot.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import config
from rag import NO_ANSWER_MESSAGE_EN, BookRAG
from vectorstore import index_exists, read_info

st.set_page_config(page_title=f"{config.BOOK_TITLE} – Book Chatbot", page_icon="📖", layout="centered")

SAMPLE_QUESTIONS = [
    "দেবদাস ও পার্ব্বতীর গ্রামের নাম কী?",
    "চন্দ্রমুখী কে?",
    "পার্ব্বতীর বিয়ে কার সঙ্গে হয়েছিল?",
    "দেবদাসের মৃত্যু কোথায় হয়?",
    "শরৎচন্দ্র চট্টোপাধ্যায় কবে জন্মগ্রহণ করেন?",  # not in the book -> no-answer demo
]


@st.cache_resource(show_spinner="এমবেডিং মডেল ও FAISS ইনডেক্স লোড হচ্ছে… (প্রথমবার একটু সময় লাগবে)")
def load_rag() -> BookRAG:
    return BookRAG()


def render_sources(sources: list[dict]) -> None:
    for s in sources:
        st.markdown(f"**[{s['number']}] {s['section']}** — [Wikisource ↗]({s['url']})")
        st.caption(s["text"])


# ------------------------------- sidebar ---------------------------------- #
with st.sidebar:
    st.header(f"📖 {config.BOOK_TITLE}")
    st.write(f"লেখক: **{config.BOOK_AUTHOR}**")
    st.markdown(f"[Bengali Wikisource-এ বইটি দেখুন]({config.BOOK_URL})")
    info = read_info()
    if info:
        st.caption(
            f"Index: {info.get('num_chunks', '?')} chunks · {info.get('num_chapters', '?')} chapters\n\n"
            f"Embedding: `{info.get('embedding_model', '?')}`\n\n"
            f"LLM: `{config.GROQ_MODEL}` (Groq)"
        )
    top_k = st.slider("Retrieved passages (top-k)", 1, 10, config.TOP_K)
    st.divider()
    st.subheader("নমুনা প্রশ্ন")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ------------------------------- main page -------------------------------- #
st.title(f"📖 {config.BOOK_TITLE} – Knowledge Base Chatbot")
st.caption(
    "বইয়ের পাঠ থেকেই উত্তর দেওয়া হয় এবং অধ্যায় উল্লেখ করা হয়। বইয়ে উত্তর না থাকলে চ্যাটবট সেটা জানিয়ে দেবে। "
    "· Answers come only from the book, with chapter citations."
)

if not index_exists():
    st.error("FAISS index পাওয়া যায়নি। প্রথমে টার্মিনালে `python ingest.py` চালান।")
    st.stop()

try:
    rag = load_rag()
except Exception as exc:  # missing API key, model mismatch, ...
    st.error(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and not msg.get("found", True):
            st.warning(msg["content"] + "\n\n" + NO_ANSWER_MESSAGE_EN)
        else:
            st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 উৎস / Sources", expanded=True):
                render_sources(msg["sources"])
        if msg.get("retrieved") and not msg.get("found", True):
            with st.expander("Retrieved passages (not relevant enough to answer)"):
                render_sources(msg["retrieved"])

question = st.chat_input("বই সম্পর্কে প্রশ্ন করুন… (Ask a question about the book)")
question = st.session_state.pop("pending_question", None) or question

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("বই থেকে খোঁজা হচ্ছে…"):
                result = rag.ask(question, k=top_k)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"⚠️ Error: {exc}", "found": True}
            )
        else:
            d = result.to_dict()
            if result.found:
                st.markdown(result.answer)
                with st.expander("📚 উৎস / Sources", expanded=True):
                    render_sources(d["sources"])
            else:
                st.warning(result.answer + "\n\n" + NO_ANSWER_MESSAGE_EN)
                with st.expander("Retrieved passages (not relevant enough to answer)"):
                    render_sources(d["retrieved"])
            st.session_state["messages"].append({
                "role": "assistant", "content": result.answer, "found": result.found,
                "sources": d["sources"], "retrieved": d["retrieved"],
            })
