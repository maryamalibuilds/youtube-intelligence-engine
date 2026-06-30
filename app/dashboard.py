"""Streamlit dashboard — the user-interaction layer.

Run:  streamlit run app/dashboard.py

Flow: enter your YouTube + OpenAI API keys in the sidebar → paste a YouTube link
in the main panel → hit Analyze. It scrapes that video's comments and runs the
whole pipeline (clean → NER/sentiment/keywords/topics → index → RAG), then the
tabs below populate.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import streamlit as st

from config import PROCESSED_DIR

st.set_page_config(page_title="YouTube Intelligence Engine", layout="wide")
ENRICHED = PROCESSED_DIR / "enriched.json"


@st.cache_data
def load_enriched():
    if ENRICHED.exists():
        return json.loads(ENRICHED.read_text(encoding="utf-8"))
    return []


@st.cache_resource
def get_orchestrator():
    from src.agent.orchestrator import Orchestrator
    from src.rag.retriever import Retriever
    from src.rag.vectorstore import VectorStore

    return Orchestrator(Retriever(VectorStore()), mode="rule")


# ---- Sidebar: API keys --------------------------------------------------------
with st.sidebar:
    st.header("🔑 API keys")
    yt_key = st.text_input("YouTube API key", type="password", key="youtube_key",
                           help="YouTube Data API v3 key from the Google Cloud console.")
    oa_key = st.text_input("OpenAI API key", type="password", key="openai_key",
                           help="Used for RAG Q&A + summarization (the Ask tab).")
    st.caption("Kept only for this browser session — never written to disk or git.")
    st.markdown(
        f"YouTube&nbsp;{'✅' if yt_key else '⬜'} &nbsp;·&nbsp; OpenAI&nbsp;{'✅' if oa_key else '⬜'}",
        unsafe_allow_html=True,
    )

# Propagate keys to the environment so the scraper + generator read them live.
if st.session_state.get("youtube_key"):
    os.environ["YOUTUBE_API_KEY"] = st.session_state["youtube_key"]
if st.session_state.get("openai_key"):
    os.environ["OPENAI_API_KEY"] = st.session_state["openai_key"]
    os.environ["LLM_PROVIDER"] = "openai"


# ---- Main panel: paste a link, analyze ---------------------------------------
st.title("🎬 YouTube Intelligence Engine")
st.subheader("Analyze a video")

c1, c2, c3 = st.columns([6, 2, 2])
url = c1.text_input("YouTube link", key="video_url",
                    placeholder="https://www.youtube.com/watch?v=…",
                    label_visibility="collapsed")
max_c = c2.number_input("Max comments", min_value=50, max_value=1000, value=300,
                        step=50, label_visibility="collapsed")
go = c3.button("🚀 Analyze", type="primary", use_container_width=True)

if go:
    if not st.session_state.get("youtube_key"):
        st.error("Add your YouTube API key in the sidebar first.")
    elif not url.strip():
        st.error("Paste a YouTube link.")
    else:
        with st.spinner("Scraping + analyzing… first run downloads models (~1–2 min)."):
            try:
                from src.pipeline import process_video

                n = len(process_video(url, int(max_c),
                                      api_key=st.session_state["youtube_key"]))
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"Analyzed {n} comments.")
                st.rerun()
            except Exception as e:  # surface the real error to the user
                st.exception(e)

data = load_enriched()
if not data:
    st.info("Enter your keys in the sidebar, paste a YouTube link above, and hit **Analyze**.")
    st.stop()

st.divider()
tab_overview, tab_ask, tab_explore = st.tabs(["📊 Overview", "💬 Ask", "🔍 Explore"])

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Comments", len(data))
    sent = Counter(d["sentiment"] for d in data)
    c2.metric("Positive", sent.get("positive", 0))
    c3.metric("Negative", sent.get("negative", 0))

    st.subheader("Sentiment distribution")
    st.bar_chart(dict(sent))

    st.subheader("Top entities")
    ents = Counter(e["text"].lower() for d in data for e in d.get("entities", []))
    st.write(ents.most_common(15) or "No entities found.")

    st.subheader("Top keywords")
    kws = Counter(k for d in data for k in d.get("keywords", []))
    st.write(kws.most_common(20))

with tab_ask:
    if not st.session_state.get("openai_key"):
        st.warning("Add your OpenAI key in the sidebar for generated answers — "
                   "retrieval/evidence still works without it.")
    st.caption("Routed by the agent to Q&A, summarization, or sentiment insight.")
    q = st.text_input("Ask about the comments", "Summarize what people think")
    if st.button("Ask", type="primary"):
        with st.spinner("Routing + retrieving…"):
            result = get_orchestrator().handle(q)
        st.success(f"Tool used: **{result.tool}**")
        st.write(result.output)
        with st.expander("Evidence (retrieved comments)"):
            for e in result.evidence:
                st.markdown(f"- {e['text']}")

with tab_explore:
    st.caption("Shows contraction/acronym expansion from the two dictionaries.")
    for d in data[:50]:
        with st.expander(d.get("text", "")[:80] or "(empty)"):
            st.markdown(f"**Raw:** {d.get('text','')}")
            st.markdown(f"**Cleaned:** {d.get('clean_text','')}")
            st.markdown(f"**Sentiment:** {d['sentiment']} ({d.get('sentiment_score','')})")
            st.markdown(f"**Entities:** {d.get('entities', [])}")
            st.markdown(f"**Keywords:** {d.get('keywords', [])}")
