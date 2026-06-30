"""Shared embedding model (singleton) for retrieval eval + semantic metrics.

The vector store uses Chroma's own embedding function for indexing; this module
exposes the same model for ad-hoc similarity (e.g. answer-relevance scoring in
evaluation) so we don't load the transformer twice with different weights.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: List[str]):
    return _model().encode(texts, normalize_embeddings=True)


def cosine(a: str, b: str) -> float:
    import numpy as np

    va, vb = embed([a, b])
    return float(np.dot(va, vb))
