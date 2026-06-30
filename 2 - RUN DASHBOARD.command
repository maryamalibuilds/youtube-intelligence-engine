#!/bin/bash
# Double-click this to open the dashboard (after you've run SETUP once).
# Your web browser will open automatically.
# To stop it later: come back to this window and press Control + C.
cd "$(dirname "$0")"
PYTHONPATH=. ./.venv/bin/streamlit run app/dashboard.py
