# ML Experiment Analytics Platform

Inspired by **AiDA Develop** — American Express’s enterprise ML platform. This repo runs XGBoost experiments on a credit card fraud dataset, logs every run to SQLite with **reproducibility metadata** (git hash, library versions, data fingerprint, random seed), classifies **failure categories**, and serves an **interactive** Streamlit dashboard (filters, compare two runs, CSV export, auto-generated insights).

**Python:** Use **3.11 or 3.12** for reliable wheels (`numpy`, `xgboost`). See `.python-version`.

## Features

| Area | Details |
|------|---------|
| **Experiments** | 27-run grid (`n_estimators`, `max_depth`, `learning_rate`) + per-stage timing (fit / predict) |
| **Logging** | SQLite `data/experiments.db`; optional `EXPERIMENTS_DB_PATH` for Docker |
| **Reproducibility** | `repro_info.py`: git commit, sklearn/xgboost/numpy/pandas versions, SHA-256 of CSV, row count, seed |
| **Failures** | `failure_category` (e.g. memory, value_error, other) for leadership-style rollups |
| **Dashboard** | Sidebar filters (min AUC, LR, depth, trees), compare A vs B, download leaderboard CSV, key insights, lineage expander |
| **Demo DB** | `deploy/sample_experiments.db` auto-copied on first load if no local DB (Streamlit Cloud / Docker) |
| **Tests** | `pytest tests/` |

## Architecture

| Layer | Role |
|--------|------|
| Data | `creditcard.csv` (via `download_data.py`) |
| Experiments | `experiments.py` |
| Tracking | `tracker.py` |
| Storage | `database.py` |
| Repro | `repro_info.py` |
| Analytics | `analytics.py` |
| Bootstrap | `bootstrap.py` |
| UI | `dashboard.py` |

## Run locally

```bash
git clone https://github.com/basva-murali-karthik/amex-ml-tracker.git
cd amex-ml-tracker
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Place **`creditcard.csv`** in the project root, or let `download_data.py` fetch it.

```bash
python experiments.py          # train + log all runs
python analytics.py            # optional CLI summary
pytest tests/                    # optional
streamlit run dashboard.py
```

## Deploy

See **[DEPLOY.md](DEPLOY.md)** for **Streamlit Community Cloud** and **Docker** (e.g. GCP Cloud Run). You need to **push the repo to GitHub** and connect it in the Streamlit Cloud UI—this cannot be automated from here.


---

*Basva Murali Karthik · BITS Pilani Goa*
