"""MLflow tracking for the pipeline.

Logs preprocessing/enrichment stats, retrieval quality, and RAG eval metrics so
you can show the "evaluation & monitoring" rubric item (15%) with real run
history and drift comparison across data pulls.

Use as a context manager:

    with track_run("enrichment", params={"n_comments": 500}) as run:
        run.log_metrics({"pos_ratio": 0.42, "avg_entities": 1.8})
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

from config import MLFLOW_TRACKING_URI


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


@contextmanager
def track_run(run_name: str, params: Optional[Dict] = None,
              experiment: str = "youtube-intelligence"):
    """Start an MLflow run; degrades to a no-op if mlflow isn't installed."""
    try:
        import mlflow
    except ImportError:
        class _Noop(_Run):
            def __init__(self): pass
            def log_metrics(self, *a, **k): pass
            def log_params(self, *a, **k): pass
            def log_dict(self, *a, **k): pass
        print("[mlflow] not installed -> metrics not persisted.")
        yield _Noop()
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        yield _Run(mlflow)
