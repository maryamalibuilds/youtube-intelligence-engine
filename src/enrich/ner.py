"""Named Entity Recognition via spaCy.

Extracts entities (people, orgs, products, GPE, etc.) from cleaned comments.
Run once:  python -m spacy download en_core_web_sm
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

_NLP = None


def _load(model: str = "en_core_web_sm"):
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load(model, disable=["lemmatizer"])
        except OSError as e:
            raise OSError(
                "spaCy model missing. Run: python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def extract(text: str) -> List[Dict]:
    doc = _load()(text)
    return [{"text": e.text, "label": e.label_} for e in doc.ents]


def extract_batch(texts: List[str], batch_size: int = 64) -> List[List[Dict]]:
    nlp = _load()
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["lemmatizer"]):
        out.append([{"text": e.text, "label": e.label_} for e in doc.ents])
    return out


def aggregate(entity_lists: List[List[Dict]], top_k: int = 25) -> Dict[str, List]:
    """Roll per-comment entities up into corpus-level counts by label."""
    by_label: Dict[str, Counter] = {}
    for ents in entity_lists:
        for e in ents:
            by_label.setdefault(e["label"], Counter())[e["text"].lower()] += 1
    return {lbl: c.most_common(top_k) for lbl, c in by_label.items()}
