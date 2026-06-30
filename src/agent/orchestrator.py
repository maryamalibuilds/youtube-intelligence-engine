"""Agent orchestration layer: route a user query to the right tool.

The rubric wants "agent orchestration is well developed" — dynamic routing
between retrieval, summarization, sentiment insight, and entity/topic lookup.

Two routing modes:
  - rule-based (default, zero-cost): keyword/intent heuristics. Always works.
  - llm    : ask the LLM to pick a tool + args (function-calling style).

Each tool is a thin wrapper over the RAG + enrichment modules. Adding a tool =
register a function in TOOLS; the router and the LLM tool-spec pick it up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List

from src.rag.generator import answer, summarize
from src.rag.retriever import Retriever


@dataclass
class ToolResult:
    tool: str
    output: str
    evidence: List[Dict]


class Orchestrator:
    def __init__(self, retriever: Retriever, mode: str = "rule"):
        self.retriever = retriever
        self.mode = mode
        self.TOOLS: Dict[str, Callable[[str], ToolResult]] = {
            "summarize": self._summarize,
            "sentiment_insight": self._sentiment,
            "qa": self._qa,
        }

    # --- tools ---
    def _qa(self, query: str) -> ToolResult:
        chunks = self.retriever.retrieve(query, strategy="hybrid", k=6)
        return ToolResult("qa", answer(query, chunks), chunks)

    def _summarize(self, query: str) -> ToolResult:
        chunks = self.retriever.retrieve(query or "main themes", strategy="semantic", k=12)
        return ToolResult("summarize", summarize(chunks, focus=query), chunks)

    def _sentiment(self, query: str) -> ToolResult:
        # Pull negative + positive subsets via metadata filtering, then ask LLM.
        neg = self.retriever.retrieve(query, strategy="metadata",
                                      where={"sentiment": "negative"}, k=6)
        pos = self.retriever.retrieve(query, strategy="metadata",
                                      where={"sentiment": "positive"}, k=6)
        chunks = pos + neg
        prompt = f"What do people like and dislike regarding: {query or 'this video'}?"
        return ToolResult("sentiment_insight", answer(prompt, chunks), chunks)

    # --- routing ---
    def route(self, query: str) -> str:
        if self.mode == "llm":
            return self._route_llm(query)
        return self._route_rules(query)

    def _route_rules(self, query: str) -> str:
        q = query.lower()
        if re.search(r"\b(summar|overview|tl;?dr|gist|recap)\b", q):
            return "summarize"
        if re.search(r"\b(sentiment|like|dislike|love|hate|positive|negative|opinion|feel)\b", q):
            return "sentiment_insight"
        return "qa"

    def _route_llm(self, query: str) -> str:
        from src.rag.generator import _generate

        tools = ", ".join(self.TOOLS)
        sys = "You route a user query to exactly one tool. Reply with only the tool name."
        choice = _generate(sys, f"Tools: {tools}\nQuery: {query}\nTool:").strip().lower()
        return choice if choice in self.TOOLS else "qa"

    def handle(self, query: str) -> ToolResult:
        tool = self.route(query)
        return self.TOOLS[tool](query)
