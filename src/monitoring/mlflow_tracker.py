"""MLflow tracking for the pipeline.

Logs preprocessing/enrichment stats, retrieval quality, and RAG eval metrics so
you can show the "evaluation & monitoring" rubric item (15%) with real run
history and drift comparison across data pulls.

Monitoring must NEVER break the core pipeline. If mlflow is missing or its
backend errors, track_run() degrades to a no-op and the analysis still runs.

Use as a context manager:

    with track_run("enrichment", params={"n_comments": 500}) as run:
        run.log_metrics({"pos_ratio": 0.42, "avg_entities": 1.8})
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Optional

from config import MLFLOW_TRACKING_URI

# MLflow 3.x raises on the local file store (./mlruns) unless you opt in. This
# project uses the simple file store, so allow it explicitly before mlflow loads.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class _Run:
    def __init__(self, mlflow):
        self._mlflow = mlflow

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            try:
                self._mlflow.log_metric(k, float(v), step=step)
            except (ValueError, TypeError):
                self._mlflow.log_param(k, v)

    def log_params(self, params: Dict):
        self._mlflow.log_params(params)

    def log_dict(self, d: Dict, artifact_file: str):
        self._mlflow.log_dict(d, artifact_file)


class _NoopRun:
    """Stand-in when mlflow is unavailable or errors — keeps the pipeline alive."""

    def log_metrics(self, *a, **k):
        pass

    def log_params(self, *a, **k):
        pass

    def log_dict(self, *a, **k):
        pass


@contextmanager
def track_run(run_name: str, params: Optional[Dict] = None,
              experiment: str = "youtube-intelligence"):
    """Start an MLflow run; fall back to a no-op on any setup failure."""
    handle = None
    active = None
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment)
        handle = mlflow.start_run(run_name=run_name)
        handle.__enter__()
        if params:
            mlflow.log_params(params)
        active = _Run(mlflow)
    except Exception as e:  # mlflow missing, backend error, etc.
        print(f"[mlflow] tracking disabled ({type(e).__name__}); pipeline continues.")
        if handle is not None:
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
            handle = None
        active = None

    # The yield is OUTSIDE the setup try/except so errors from the pipeline body
    # propagate normally (we only swallow *tracking* failures, never real work).
    try:
        yield active if active is not None else _NoopRun()
    finally:
        if handle is not None:
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
