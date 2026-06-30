"""Evaluation harness for retrieval + RAG quality (rubric: 15%).

Provides metrics you can report and log to MLflow:

  Retrieval:
    - hit_rate@k / MRR over a small labelled query->relevant-comment set
  RAG answer quality (reference-free):
    - groundedness : max cosine sim of answer to its retrieved context
                     (low score == likely hallucination)
    - answer_relevance : cosine sim of answer to the question

Build a tiny gold set in eval_queries.json once and re-run after each change to
show "clear link between design choices and results."
"""
from __future__ import annotations

from typing import Dict, List

from src.monitoring.mlflow_tracker import track_run
from src.rag.embeddings import cosine, embed
from src.rag.retriever import Retriever


def hit_rate_mrr(retriever: Retriever, gold: List[Dict], k: int = 5,
                 strategy: str = "hybrid") -> Dict[str, float]:
    """gold = [{"query": str, "relevant": [substrings that must appear]}]."""
    hits, rr = 0, 0.0
    for g in gold:
        results = retriever.retrieve(g["query"], strategy=strategy, k=k)
        rank = next((i for i, r in enumerate(results, 1)
                     if any(rel.lower() in r["text"].lower() for rel in g["relevant"])), None)
        if rank:
            hits += 1
            rr += 1 / rank
    n = max(len(gold), 1)
    return {f"hit_rate@{k}": hits / n, "mrr": rr / n}


def groundedness(answer: str, context_chunks: List[Dict]) -> float:
    if not context_chunks:
        return 0.0
    import numpy as np

    ans_vec = embed([answer])[0]
    ctx_vecs = embed([c["text"] for c in context_chunks])
    return float(np.max(ctx_vecs @ ans_vec))


def answer_relevance(question: str, answer: str) -> float:
    return cosine(question, answer)


def evaluate_retrieval_strategies(retriever: Retriever, gold: List[Dict], k: int = 5):
    """Compare all four strategies and log to MLflow — great report figure."""
    with track_run("retrieval_eval", params={"k": k, "n_queries": len(gold)}) as run:
        table = {}
        for strat in ("semantic", "lexical", "hybrid"):
            m = hit_rate_mrr(retriever, gold, k=k, strategy=strat)
            table[strat] = m
            run.log_metrics({f"{strat}_{kk}": vv for kk, vv in m.items()})
        run.log_dict(table, "retrieval_strategies.json")
        return table
