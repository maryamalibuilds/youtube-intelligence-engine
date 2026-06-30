# Handoff — read this first

Hey 👋 This is the **CSCI370 "YouTube Intelligence Engine"** project, already
scaffolded end-to-end. You can run it in ~10 minutes and start filling in the
real work. This doc tells you what exists, what's left, and where the marks are.

## What this project is (60-second version)

Scrape YouTube comments → clean the noisy slang → extract insight (entities,
keywords, sentiment, topics) → index into a vector DB → answer questions /
summarize with an LLM (RAG) → route queries with an agent → monitor with MLflow
→ show it in a Streamlit dashboard. Due **week 10**, demo **week 11** (10 min
demo + 5 min Q&A). Full spec + rubric is in the course doc; report outline is in
`docs/REPORT.md`.

## Current state

| Part | State |
|------|-------|
| Project structure, config, requirements | ✅ done |
| Preprocessing (`src/preprocess/`) — uses the 2 dictionaries | ✅ **done + unit-tested** |
| Scraper, enrichment, RAG, agent, MLflow, dashboard | ✅ implemented, runs offline on a built-in sample |
| Real dataset, evaluation gold set, the written report | ⬜ **your job** |

Everything **runs without API keys** on a tiny demo sample, so nothing is
blocked. Add keys to scrape real data and enable LLM answers.

## Get it running

```bash
cd youtube-intelligence-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env          # add YOUTUBE_API_KEY + an LLM key (optional to start)

python -m src.pipeline        # offline demo: scrape→clean→enrich→index
streamlit run app/dashboard.py
pytest -q                     # preprocessing tests should pass
```

## Where the grade actually is (do these well)

1. **RAG & LLM — 20%** → `src/rag/`. Four retrieval strategies + hybrid fusion
   are already coded. Your job: pick a real topic, tune prompts, show good Q&A.
2. **Evaluation & Monitoring — 15%** → `src/monitoring/`. Build a small
   `data/eval_queries.json` gold set, run `evaluate.evaluate_retrieval_strategies`,
   screenshot the MLflow runs. Do an ablation (sentiment with vs. without slang
   expansion) — easy, high-value.
3. Dataset (10%), insight extraction (10%), topic modeling (10%) → `src/enrich/`.

## Suggested split (if 2 people)

- **Person A:** scraping + preprocessing tuning + enrichment (NER/sentiment/
  keywords/topics) + the report sections 3–4.
- **Person B:** RAG + agent + evaluation/MLflow + dashboard polish + report
  sections 2, 5, 6.
- Both: intro/conclusion + the demo.

## Don't forget (rubric requires it)

- GitHub repo with this README **showing iteration** → commit per stage, don't
  dump it all in one commit.
- The written report (`docs/REPORT.md` is a fill-in-the-blanks scaffold).
- Sample inputs/outputs + screenshots in the report appendix.

Module-by-module detail is in `README.md`.
