"""Central configuration. Reads .env once and exposes typed settings.

Everything downstream imports from here so there are no scattered os.getenv calls.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    # Allow stdlib-only modules (e.g. the normalizer) to run before deps are
    # installed. Real env vars still work; only .env auto-loading is skipped.
    pass

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CONTRACTIONS_FILE = DATA_DIR / "contractions.txt"
ACRONYMS_FILE = DATA_DIR / "acronyms.csv"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- API keys ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- LLM ---
# Default provider is Groq: free, fast, OpenAI-compatible. Override via env.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty -> generator picks a per-provider default

# --- Models / stores ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", str(PROCESSED_DIR / "chroma"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def has_youtube() -> bool:
    return bool(YOUTUBE_API_KEY)


def has_llm() -> bool:
    return bool(GROQ_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY) or LLM_PROVIDER == "local"
