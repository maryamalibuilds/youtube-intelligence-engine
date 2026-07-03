# YouTube Intelligence Engine

End-to-end NLP pipeline that scrapes YouTube comments, cleans noisy user text,
extracts insight (NER, keywords, sentiment, topics), and serves grounded Q&A +
summarization through a **RAG** system with an **agent** routing layer and
**MLflow** monitoring.

> CSCI370 course project. Pipeline: scrape → preprocess → enrich → index → RAG → agent → dashboard, monitored with MLflow.

## Architecture

```
                         ┌─────────────────────────────────────────┐
 YouTube Data API  ──▶   │  scrape  →  preprocess  →  enrich        │
   (or offline sample)   │   raw       clean text     NER           │
                         │             (contractions  sentiment     │
                         │              + acronyms)    keywords      │
                         │                             topics        │
                         └───────────────┬─────────────────────────┘
                                         ▼
                          ┌──────────────────────────────┐
                          │   Vector store (ChromaDB)     │
                          │   text + embedding + metadata │
                          └───────────────┬──────────────┘
                                          ▼
            ┌──────────────────────────────────────────────────┐
            │  RAG:  retrieval (semantic│lexical│metadata│hybrid)│
            │        + LLM (Q&A / summarization, grounded)       │
            └───────────────┬──────────────────────────────────┘
                            ▼
                ┌────────────────────────┐      ┌──────────────┐
                │  Agent orchestrator     │ ◀──▶ │   MLflow      │
                │  routes query → tool    │      │  monitoring   │
                └───────────┬────────────┘      └──────────────┘
                            ▼
                   Streamlit dashboard
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # for NER

cp .env.example .env                         # then add your keys
```

`.env` keys (all optional — the pipeline runs offline on a built-in sample if
they are blank):

| Key | Used for |
|-----|----------|
| `YOUTUBE_API_KEY` | scraping real comments (YouTube Data API v3) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | RAG generation + LLM routing |

## Run it

```bash
# 1. Scrape + clean + enrich + index (offline demo if no API key)
python -m src.pipeline --query "your topic here" --videos 3 --max 300

# 2. Inspect dictionary-based cleaning in isolation
python -m src.preprocess.normalizer

# 3. Launch the dashboard
streamlit run app/dashboard.py

# 4. View MLflow runs
mlflow ui --backend-store-uri ./mlruns

# 5. Tests
pytest -q
```

## Preprocessing dictionaries

Two normalization dictionaries live in `data/` and are applied in
`src/preprocess/normalizer.py`:

- **`contractions.txt`** (`key:value`) — `don't → do not`, `y'all → you all`
- **`acronyms.csv`** (`key,value`) — `idk → I don't know`, `gr8 → great`,
  `2moro → tomorrow` (3000+ internet-slang entries)

Contractions win over acronyms on key collisions; multi-word keys
(`fo shizzle`) are matched before single tokens. See `tests/test_normalizer.py`.

## Module map

| Path | Responsibility |
|------|----------------|
| `src/scrape/youtube_scraper.py` | YouTube Data API v3 comment scraping |
| `src/preprocess/` | `normalizer` (dict expansion + spell), `cleaner` (full pipeline) |
| `src/enrich/` | `ner`, `keywords`, `sentiment`, `topics` |
| `src/rag/` | `vectorstore`, `retriever` (4 strategies), `generator`, `embeddings` |
| `src/agent/orchestrator.py` | query → tool routing (rule-based + LLM) |
| `src/monitoring/` | `mlflow_tracker`, `evaluate` (retrieval + RAG metrics) |
| `src/pipeline.py` | one command runs the whole flow |
| `app/dashboard.py` | Streamlit UI |

## Rubric coverage

| Rubric item | Where |
|-------------|-------|
| Dataset construction | `scrape/` + `preprocess/` |
| NER / keyword / sentiment | `enrich/ner,keywords,sentiment` |
| Topic modeling (general + per-sentiment) | `enrich/topics` |
| RAG & LLM (20%) | `rag/` — 4 retrieval strategies, hybrid fusion, grounded prompts |
| Agent orchestration | `agent/orchestrator` |
| Evaluation & monitoring (15%) | `monitoring/evaluate` + MLflow |
| Dashboard | `app/dashboard` |

## Evidence / Screenshots

## Evidence / Screenshots

### Dataset Construction

The final dataset contains 10,764 raw comments, 10,764 enriched comments and 63 unique videos.

![Dataset proof](./1-10k%20data%20set%20scraping.png)

---

### Dashboard Overview

The Streamlit dashboard displays the final dataset size and sentiment distribution.

![Dashboard overview](./1%29dashboard.png)

---

### Topic Modelling

Initial topic modelling was noisy because YouTube comments contained emoji-related tokens and repeated informal words.

![Initial topic modelling](./2-topic%20modeling.png)

The topic modelling was then improved with extra cleaning and stopwords.

![Cleaner topic modelling](./3-%20cleaner%20topic%20modelling.png)

The final cleaned topic modelling results were used in the report.

![Final topic modelling](./4-cleanest%20topic%20modelling.png)

---

### Retrieval Evaluation

Semantic, lexical and hybrid retrieval strategies were evaluated using hit rate@5 and MRR.

![Retrieval evaluation](./5-results.png)

---

### MLflow Monitoring

MLflow was used to track pipeline and retrieval evaluation runs.

![MLflow runs](./7-ml%20run.png)

The hybrid retrieval run recorded hit_rate_at_5 = 0.75 and MRR = 0.75.

![MLflow hybrid metrics](./8-ml%20hybrid%20metrics.png)
