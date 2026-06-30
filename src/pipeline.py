"""End-to-end pipeline: scrape -> clean -> enrich -> index.

Run the whole thing:
    python -m src.pipeline --query "react native tutorial" --videos 3 --max 300
    python -m src.pipeline --video dQw4w9WgXcQ --max 500
    python -m src.pipeline                      # offline demo sample

Produces:
    data/raw/<out>.json          raw scraped comments
    data/processed/enriched.json cleaned + NER + sentiment + keywords
    data/processed/chroma/       vector index (ready for RAG)
    mlruns/                      MLflow metrics for this run
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from config import PROCESSED_DIR
from src.monitoring.mlflow_tracker import track_run
from src.preprocess.cleaner import clean_batch
from src.scrape.youtube_scraper import fetch_comments, save, search_videos


def enrich(comments: List[Dict]) -> List[Dict]:
    """Clean each comment and attach sentiment, entities, keywords."""
    from src.enrich import keywords, ner, sentiment

    raw_texts = [c["text"] for c in comments]
    cleaned = clean_batch(raw_texts)

    sentiments = sentiment.analyze_batch(cleaned)
    try:
        entity_lists = ner.extract_batch(cleaned)
    except OSError as e:
        print(f"[enrich] NER skipped: {e}")
        entity_lists = [[] for _ in cleaned]

    enriched = []
    for c, clean, sent, ents in zip(comments, cleaned, sentiments, entity_lists):
        enriched.append({
            **c,
            "clean_text": clean,
            "sentiment": sent["label"],
            "sentiment_score": sent["score"],
            "entities": ents,
            "keywords": [k for k, _ in keywords.extract(clean, top_k=5)],
        })
    return enriched


def index(enriched: List[Dict]):
    """Push enriched comments into the vector store."""
    from src.rag.vectorstore import VectorStore

    store = VectorStore()
    ids = [e.get("comment_id", str(i)) for i, e in enumerate(enriched)]
    docs = [e["clean_text"] for e in enriched]
    metas = [{
        "video_id": e.get("video_id", ""),
        "author": e.get("author", ""),
        "likes": e.get("likes", 0),
        "sentiment": e["sentiment"],
        "entities": [ent["text"] for ent in e["entities"]],
        "keywords": e["keywords"],
    } for e in enriched]
    # De-dup ids defensively (Chroma requires unique ids).
    seen, uids = set(), []
    for i, _id in enumerate(ids):
        uid = _id if _id not in seen else f"{_id}-{i}"
        seen.add(uid)
        uids.append(uid)
    store.add(uids, docs, metas)
    return store.count()


def run(comments: List[Dict], out: str = "comments"):
    save(comments, out)
    with track_run("pipeline", params={"n_comments": len(comments)}) as run_:
        enriched = enrich(comments)
        sent_dist = Counter(e["sentiment"] for e in enriched)
        total = max(len(enriched), 1)
        run_.log_metrics({f"sentiment_{k}_ratio": v / total for k, v in sent_dist.items()})
        run_.log_metrics({"avg_entities": sum(len(e["entities"]) for e in enriched) / total})

        out_path = PROCESSED_DIR / "enriched.json"
        out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[pipeline] enriched -> {out_path}  (sentiment: {dict(sent_dist)})")

        n = index(enriched)
        run_.log_metrics({"indexed_docs": n})
        print(f"[pipeline] indexed {n} docs into vector store.")
    return enriched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--query")
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--out", default="comments")
    args = ap.parse_args()

    comments: List[Dict] = []
    if args.video:
        comments = fetch_comments(args.video, args.max)
    elif args.query:
        for vid in search_videos(args.query, args.videos):
            comments += fetch_comments(vid, args.max)
    else:
        comments = fetch_comments("demo", args.max)

    run(comments, args.out)


if __name__ == "__main__":
    main()
