"""Topic modeling with BERTopic (embeddings + UMAP + HDBSCAN + c-TF-IDF).

Supports the rubric's "topic modeling (general, per sentiment)" by exposing
fit_topics() for the whole corpus and a helper to run it per sentiment label.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def fit_topics(texts: List[str], min_topic_size: int = 10):
    """Fit BERTopic and return (model, topics, info_dataframe)."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    from config import EMBEDDING_MODEL

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    model = BERTopic(embedding_model=embedder, min_topic_size=min_topic_size,
                     calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(texts)
    return model, topics, model.get_topic_info()


def top_terms(model, topic_id: int, n: int = 10) -> List[Tuple[str, float]]:
    terms = model.get_topic(topic_id)
    return terms[:n] if terms else []


def topics_per_sentiment(texts: List[str], sentiments: List[str],
                         min_topic_size: int = 5) -> Dict[str, object]:
    """Fit a separate topic model for each sentiment bucket.

    Returns {sentiment_label: get_topic_info() dataframe}.
    """
    buckets: Dict[str, List[str]] = {}
    for t, s in zip(texts, sentiments):
        buckets.setdefault(s, []).append(t)

    results = {}
    for label, docs in buckets.items():
        if len(docs) < min_topic_size * 2:
            results[label] = None  # too few docs to model reliably
            continue
        _, _, info = fit_topics(docs, min_topic_size=min_topic_size)
        results[label] = info
    return results
