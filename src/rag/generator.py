"""LLM generation for grounded Q&A and summarization.

Provider-agnostic, with keys + model read at CALL TIME from the environment so a
key typed into the dashboard takes effect immediately. Default provider is
**Groq** — free, fast, and OpenAI-API-compatible, so we reuse the OpenAI SDK
pointed at Groq's endpoint. OpenAI and Anthropic also work via LLM_PROVIDER.

Uses structured prompts that inject retrieved comments as context and instruct
the model to stay grounded (cite, don't hallucinate).
"""
from __future__ import annotations

import os
from typing import Dict, List

from config import (ANTHROPIC_API_KEY, GROQ_API_KEY, LLM_MODEL, LLM_PROVIDER,
                    OPENAI_API_KEY)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}

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
def _provider() -> str:
    return (os.getenv("LLM_PROVIDER") or LLM_PROVIDER or "groq").lower()


def _model() -> str:
    return (os.getenv("LLM_MODEL") or LLM_MODEL
            or _DEFAULT_MODELS.get(_provider(), "llama-3.3-70b-versatile"))


def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY") or GROQ_API_KEY


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY


def _anthropic_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY


def _has_llm() -> bool:
    return {
        "groq": bool(_groq_key()),
        "openai": bool(_openai_key()),
        "anthropic": bool(_anthropic_key()),
        "local": True,
    }.get(_provider(), False)


def _format_context(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        tag = f" (sentiment={meta.get('sentiment')}, likes={meta.get('likes')})" if meta else ""
        lines.append(f"[{i}]{tag} {c['text']}")
    return "\n".join(lines)


def _call_openai_compatible(system: str, user: str, *, api_key: str, model: str,
                            base_url: str = None) -> str:
    """Works for both OpenAI and Groq (Groq exposes an OpenAI-compatible API)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
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
        return ("[no LLM key configured] Add your free Groq key (sidebar or .env) "
                "to enable generated answers. Retrieved context was:\n" + user[:800])
    provider = _provider()
    if provider == "groq":
        return _call_openai_compatible(system, user, api_key=_groq_key(),
                                       model=_model(), base_url=GROQ_BASE_URL)
    if provider == "anthropic":
        return _call_anthropic(system, user)
    return _call_openai_compatible(system, user, api_key=_openai_key(), model=_model())


def answer(question: str, chunks: List[Dict]) -> str:
    context = _format_context(chunks)
    user = f"Comments:\n{context}\n\nQuestion: {question}\n\nGrounded answer:"
    return _generate(QA_SYSTEM, user)


def summarize(chunks: List[Dict], focus: str = "") -> str:
    context = _format_context(chunks)
    extra = f"\nFocus the summary on: {focus}" if focus else ""
    user = f"Comments:\n{context}{extra}\n\nSummary:"
    return _generate(SUMMARY_SYSTEM, user)
