# 🛡️ Fraud Detection MLOps Pipeline

A **self-healing ML pipeline** for real-time fraud detection. Monitors itself in production, detects data drift, and automatically retrains — without human intervention.

**Live API:** https://fraud-mlops-api.onrender.com/docs

---

## Architecture

```
[IEEE-CIS Dataset — 590K transactions]
              │
              ▼
     ┌─────────────────┐
     │   src/train.py  │  XGBoost · AUC-ROC 0.946 · SHAP explanations
     └────────┬────────┘
              │ registers model
              ▼
     ┌─────────────────┐         ┌──────────────────────┐
     │  MLflow Registry│────────▶│  FastAPI  :8000      │
     │  fraud-detector │         │  POST /predict       │
     │  stage:Production         │  GET  /drift-report  │
     └─────────────────┘         └──────────┬───────────┘
              ▲                             │ metrics
              │ promote                     ▼
     ┌────────┴────────┐         ┌──────────────────────┐
     │  Airflow DAG    │         │ Prometheus + Grafana  │
     │  runs daily 2am │         │ 6-panel dashboard     │
     └────────┬────────┘         └──────────────────────┘
              │
    ┌─────────▼──────────┐
    │   check_drift      │  PSI > 0.2?
    └────────┬───────────┘
             │
      ┌──────┴──────┐
    YES            NO
      │              │
 retrain_model   skip_retrain
 promote_model
 update_ref_stats
```

---

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.9464** |
| AUC-PR | 0.6903 |
| Prediction latency p99 | **< 50ms** |
| Dataset | IEEE-CIS (590K transactions, 3.5% fraud) |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Model | XGBoost + SHAP explainability |
| API | FastAPI + Pydantic + Prometheus |
| Experiment tracking | MLflow Model Registry |
| Drift detection | PSI + Evidently AI |
| Orchestration | Apache Airflow (daily DAG) |
| Monitoring | Prometheus + Grafana (6 panels) |
| CI/CD | GitHub Actions + Render |
| Containerisation | Docker + docker-compose |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Srishh1/fraud-mlops-pipeline
cd fraud-mlops-pipeline
pip install -r requirements.txt

# 2. Download IEEE-CIS dataset from Kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/

# 3. Train model (logs to MLflow, promotes to Production)
python src/train.py

# 4. Start API
uvicorn src.api:app --reload
# → http://localhost:8000/docs

# 5. Run drift detection demo
python src/drift_detector.py

# 6. Full stack (API + MLflow + Prometheus + Grafana)
docker-compose up -d
# → Grafana:  http://localhost:3000  (admin/frauddetect123)
# → MLflow:   http://localhost:5000
# → Airflow:  http://localhost:8080  (admin/admin)
```

---

## Self-Healing Demo

```bash
# The Airflow DAG runs daily. To trigger manually:
export AIRFLOW_HOME=$(pwd)/airflow
export PYTHONPATH=$(pwd)
airflow scheduler &
airflow dags unpause fraud_detection_pipeline
airflow dags trigger fraud_detection_pipeline
# → Open http://localhost:8080 to watch tasks execute
```

**What happens:**
1. `ingest_data` loads latest transactions
2. `engineer_features` applies same transforms as training
3. `check_drift` computes PSI for 15 features
4. `branch_on_drift` — PSI > 0.2 → retrain, else → skip
5. If retraining: new model registered in MLflow → promoted to Production
6. API serves new model automatically on next request

---

## API Usage

```bash
curl -X POST https://fraud-mlops-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 500, "TransactionDT": 10800, "card4": 1, "card6": 0, "D1": 0, "dist1": 500}'
```

Response:
```json
{
  "fraud_score": 0.8613,
  "risk_level": "CRITICAL",
  "risk_action": "Automatically decline.",
  "top_reasons": [
    "Transaction amount ($500.00) increases fraud risk (+0.274)",
    "Hour of transaction (03:00) increases fraud risk (+0.039)"
  ],
  "latency_ms": 49.34,
  "model_version": "v1-Production"
}
```

---

## Project Structure

```
fraud-mlops-pipeline/
├── src/
│   ├── train.py          # XGBoost training + MLflow logging
│   ├── explainer.py      # SHAP explainability module
│   ├── api.py            # FastAPI server
│   └── drift_detector.py # PSI + Evidently drift engine
├── dags/
│   └── fraud_pipeline.py # Airflow DAG (8 tasks)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_dashboard.json
├── tests/
│   ├── test_drift.py     # 30 PSI tests
│   └── test_api.py       # API endpoint tests
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory data analysis
├── Dockerfile
├── docker-compose.yml
└── render.yaml           # Render deployment config
```

