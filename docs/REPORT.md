# YouTube Intelligence Engine — Project Report

> Fill each section. Headings match the required 8-section structure from the
> spec. Notes in _italics_ tell you what the rubric rewards — delete them before
> submitting.

---

## 1. Introduction
_Overview, motivation, objectives, significance, scope, and a bullet list of
your main contributions._

- **Problem:** turning noisy YouTube comments into actionable insight + grounded Q&A.
- **Objectives:** (1) robust preprocessing of slang-heavy text, (2) insight
  extraction, (3) a RAG system with agent routing, (4) evaluation & monitoring.
- **Scope:** comments only (not video/audio); English; topic = _<your topic>_.

## 2. Background and Related Work
_Rubric (5%): clear, well-motivated problem._
- Problem definition & motivation.
- Existing approaches: classic sentiment dashboards vs. RAG-based QA; tools like
  VADER/TextBlob, BERTopic, LangChain/LlamaIndex RAG stacks.
- Project goals & expected outcomes.
- How this project differs/improves (agent routing + per-sentiment topics +
  slang normalization dictionaries).

## 3. Methodology
_Rubric: System Design & Justification (10%), Dataset Construction (10%).
Include the architecture diagram from the README and a data-flow flowchart._
- **System architecture** (insert diagram).
- **Dataset:** topic chosen, videos/queries used, # comments, time window,
  class balance. Justify the selection.
- **Preprocessing:** contraction + acronym expansion (the two dictionaries),
  emoji handling, repeat dampening, optional spell-correction. _Show a before/
  after example table — easy marks._
- **Tools/libraries:** spaCy, transformers (twitter-roberta), KeyBERT, BERTopic,
  ChromaDB, sentence-transformers, MLflow, Streamlit.
- **Justification** for each major choice (why twitter-roberta over VADER, why
  hybrid retrieval, why ChromaDB).

## 4. System Implementation and Analytical Components
_Rubric: NER/Keyword/Sentiment (10%), Topic Modeling (10%), RAG & LLM (20%).
Screenshots + sample outputs for each._
- **NER** — sample entities table, corpus aggregation.
- **Keyword extraction** — top corpus keywords / word cloud.
- **Sentiment** — distribution chart, example classifications.
- **Topic modeling** — topics (general) + topics per sentiment, interpretation.
- **RAG** — vector representation, the 4 retrieval strategies, hybrid fusion,
  prompt design (show the system prompts), sample Q&A + summary outputs.
- **Agent orchestration** — routing logic, example of a query being routed.

## 5. Evaluation and Monitoring
_Rubric (15%): metrics + critical analysis, not just numbers._
- **Retrieval:** hit_rate@k and MRR across semantic/lexical/hybrid (table +
  which won and why). _From `monitoring/evaluate.py`._
- **RAG answers:** groundedness + answer-relevance scores; an ablation
  (sentiment F1 with vs. without acronym expansion).
- **Monitoring:** MLflow screenshots; drift across two data pulls.
- **Strengths & weaknesses** observed.

## 6. Challenges and Limitations
- API quota limits, slang the dictionaries miss, sarcasm in sentiment,
  hallucination risk, small-corpus topic instability. Support with examples.

## 7. Conclusion
- Achievements, key findings, future work (multilingual, video transcript
  fusion, fine-tuned reranker).

## 8. Appendix
- Extra figures/screenshots, sample I/O, config (`.env` keys, model names),
  setup instructions (link README), code snippets.

---

### Demo plan (week 11 — 10 min demo + 5 min Q&A)
1. (1 min) Problem + architecture slide.
2. (2 min) Run `python -m src.pipeline` live → show enriched.json + sentiment split.
3. (1 min) `normalizer.py` before/after on a gnarly comment (the two dictionaries).
4. (3 min) Dashboard: Overview → Ask (route to summarize + Q&A) → Explore.
5. (2 min) MLflow runs + retrieval-strategy comparison.
6. (1 min) Limitations + future work.
```
