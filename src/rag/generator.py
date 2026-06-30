"""LLM generation for grounded Q&A and summarization.

Provider-agnostic: routes to OpenAI, Anthropic, or a local stub. Keys + model
are read at CALL TIME from the environment (falling back to config), so a key
typed into the dashboard — which sets os.environ — takes effect immediately
without a restart. Uses structured prompts that inject retrieved comments as
context and instruct the model to stay grounded (cite, don't hallucinate).
"""
from __future__ import annotations

import os
from typing import Dict, List

from config import (ANTHROPIC_API_KEY, LLM_MODEL, LLM_PROVIDER, OPENAI_API_KEY)

QA_SYSTEM = (
    "You are an analyst answering questions about a corpus of YouTube comments. "
    "Answer ONLY from the provided comments. If they don't contain the answer, "
    "say so plainly. Quote short snippets as evidence. Be concise and neutral."
)
SUMMARY_SYSTEM = (
    "You summarize YouTube comment sections. Produce a faithful, balanced "
    "summary grounded in the provided comments: capture the main themes, the "
    "split of positive vs. negative opinion, and any recurring complaints or "
    "praise. Do not invent details not present in the comments."
)


# --- runtime settings (env overrides config; lets the dashboard inject keys) --
def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY


def _anthropic_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY


def _provider() -> str:
    return (os.getenv("LLM_PROVIDER") or LLM_PROVIDER or "openai").lower()


def _model() -> str:
    return os.getenv("LLM_MODEL") or LLM_MODEL


def _has_llm() -> bool:
    return bool(_openai_key() or _anthropic_key()) or _provider() == "local"


def _format_context(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        tag = f" (sentiment={meta.get('sentiment')}, likes={meta.get('likes')})" if meta else ""
        lines.append(f"[{i}]{tag} {c['text']}")
    return "\n".join(lines)


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_openai_key())
    resp = client.chat.completions.create(
        model=_model(),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def _call_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_anthropic_key())
    model = _model() if "claude" in _model() else "claude-3-5-haiku-latest"
    resp = client.messages.create(
        model=model, max_tokens=1024, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _generate(system: str, user: str) -> str:
    if not _has_llm():
        return ("[no LLM key configured] Add your OpenAI key (sidebar or .env) to "
                "enable generated answers. Retrieved context was:\n" + user[:800])
    provider = _provider()
    if provider == "anthropic" or (not _openai_key() and _anthropic_key()):
        return _call_anthropic(system, user)
    return _call_openai(system, user)


def answer(question: str, chunks: List[Dict]) -> str:
    context = _format_context(chunks)
    user = f"Comments:\n{context}\n\nQuestion: {question}\n\nGrounded answer:"
    return _generate(QA_SYSTEM, user)


def summarize(chunks: List[Dict], focus: str = "") -> str:
    context = _format_context(chunks)
    extra = f"\nFocus the summary on: {focus}" if focus else ""
    user = f"Comments:\n{context}{extra}\n\nSummary:"
    return _generate(SUMMARY_SYSTEM, user)
