.PHONY: install setup run dash mlflow test clean

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

setup: install
	cp -n .env.example .env || true

run:            ## scrape+clean+enrich+index (offline demo if no API key)
	python -m src.pipeline $(ARGS)

dash:
	streamlit run app/dashboard.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns

test:
	pytest -q

clean:
	rm -rf data/processed/* mlruns __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
