"""Keyword / keyphrase extraction.

Primary: KeyBERT (embedding-based, contextual). Fallback: YAKE (statistical,
no heavy deps). Both return [(phrase, score), ...] sorted by relevance.
"""
from __future__ import annotations

from typing import List, Tuple

_KB = None


def _keybert(text: str, top_k: int) -> List[Tuple[str, float]]:
    global _KB
    if _KB is None:
        from keybert import KeyBERT

        from config import EMBEDDING_MODEL

        _KB = KeyBERT(model=EMBEDDING_MODEL)
    return _KB.extract_keywords(text, keyphrase_ngram_range=(1, 2),
                                stop_words="english", top_n=top_k)


def _yake(text: str, top_k: int) -> List[Tuple[str, float]]:
    import yake

    kw = yake.KeywordExtractor(n=2, top=top_k)
    # YAKE: lower score = more relevant; invert so higher = better, like KeyBERT.
    return [(p, round(1 - s, 4)) for p, s in kw.extract_keywords(text)]


def extract(text: str, top_k: int = 8) -> List[Tuple[str, float]]:
    if not text.strip():
        return []
    try:
        return _keybert(text, top_k)
    except Exception:
        return _yake(text, top_k)


def corpus_keywords(texts: List[str], top_k: int = 30) -> List[Tuple[str, float]]:
    """Keywords over the whole corpus (joins docs; good for a word-cloud)."""
    joined = " ".join(texts)
    return extract(joined, top_k)
