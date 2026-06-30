"""Sentiment analysis tuned for short social text.

Primary: cardiffnlp/twitter-roberta-base-sentiment-latest (trained on tweets,
ideal for YouTube comments). Falls back to NLTK VADER if transformers/torch are
unavailable, so the pipeline never hard-fails.

Returns label in {negative, neutral, positive} + confidence score.
"""
from __future__ import annotations

from typing import Dict, List

from config import SENTIMENT_MODEL

_PIPE = None
_VADER = None


def _load_transformer():
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline

        _PIPE = pipeline("sentiment-analysis", model=SENTIMENT_MODEL,
                         top_k=None, truncation=True)
    return _PIPE


def _load_vader():
    global _VADER
    if _VADER is None:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            _VADER = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _VADER = SentimentIntensityAnalyzer()
    return _VADER


def _vader_label(text: str) -> Dict:
    s = _load_vader().polarity_scores(text)["compound"]
    label = "positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral"
    return {"label": label, "score": abs(s)}


def analyze(text: str) -> Dict:
    """Sentiment for a single (already-cleaned) comment."""
    try:
        scores = _load_transformer()(text)[0]
        best = max(scores, key=lambda d: d["score"])
        return {"label": best["label"].lower(), "score": round(best["score"], 4)}
    except Exception:
        return _vader_label(text)


def analyze_batch(texts: List[str]) -> List[Dict]:
    try:
        pipe = _load_transformer()
        results = pipe(list(texts))
        out = []
        for scores in results:
            best = max(scores, key=lambda d: d["score"])
            out.append({"label": best["label"].lower(), "score": round(best["score"], 4)})
        return out
    except Exception:
        return [_vader_label(t) for t in texts]
