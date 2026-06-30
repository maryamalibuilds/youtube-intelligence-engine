"""Streamlit dashboard — the user-interaction layer (rubric deliverable).

Run:  streamlit run app/dashboard.py

Tabs:
  1. Overview   — sentiment split, top entities/keywords, comment volume
  2. Ask        — agent-routed Q&A / summarization over the indexed corpus
  3. Explore    — raw vs. cleaned comment inspector (shows your dict expansion)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

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


st.title("🎬 YouTube Intelligence Engine")
data = load_enriched()

if not data:
    st.warning("No processed data yet. Run:  `python -m src.pipeline` first.")
    st.stop()

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
    st.write(ents.most_common(15) or "No entities (install spaCy model).")

    st.subheader("Top keywords")
    kws = Counter(k for d in data for k in d.get("keywords", []))
    st.write(kws.most_common(20))

with tab_ask:
    st.caption("Routed by the agent to Q&A, summarization, or sentiment insight.")
    q = st.text_input("Ask about the comments", "Summarize what people think")
    if st.button("Ask", type="primary"):
        with st.spinner("Routing + retrieving..."):
            result = get_orchestrator().handle(q)
        st.success(f"Tool used: **{result.tool}**")
        st.write(result.output)
        with st.expander("Evidence (retrieved comments)"):
            for e in result.evidence:
                st.markdown(f"- {e['text']}")

with tab_explore:
    st.caption("Shows contraction/acronym expansion from your two dictionaries.")
    for d in data[:50]:
        with st.expander(d.get("text", "")[:80] or "(empty)"):
            st.markdown(f"**Raw:** {d.get('text','')}")
            st.markdown(f"**Cleaned:** {d.get('clean_text','')}")
            st.markdown(f"**Sentiment:** {d['sentiment']} ({d.get('sentiment_score','')})")
            st.markdown(f"**Entities:** {d.get('entities', [])}")
            st.markdown(f"**Keywords:** {d.get('keywords', [])}")
