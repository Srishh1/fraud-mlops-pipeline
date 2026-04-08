# 🛡️ Fraud Detection MLOps Pipeline

A **self-healing ML pipeline** for real-time fraud detection that monitors itself in production, detects data drift, and automatically retrains — without human intervention.

Built as a portfolio project targeting **MLOps Engineer** and **FinTech AI** roles at companies like Razorpay, PhonePe, CRED, and Zepto.

---

## 🏗️ Architecture

```
[Kaggle IEEE-CIS Data]
         │
         ▼
  ┌─────────────┐
  │  train.py   │ ← XGBoost + SHAP
  └──────┬──────┘
         │ model artifact
         ▼
  ┌─────────────┐     ┌──────────────────┐
  │   MLflow    │────▶│  FastAPI Server  │ ← /predict, /drift-report
  │  Registry   │     └────────┬─────────┘
  └─────────────┘              │ metrics
         ▲                     ▼
         │              ┌─────────────┐
  ┌──────┴──────┐       │ Prometheus  │
  │   Airflow   │       │  + Grafana  │
  │  DAG (daily)│       └─────────────┘
  └─────────────┘
    │         │
    ▼         ▼
 retrain   skip
 (if drift) (if stable)
```

---

## 📁 Project Structure

```
fraud-mlops-pipeline/
├── .github/workflows/ci.yml     # GitHub Actions CI/CD
├── dags/fraud_pipeline.py       # Airflow DAG (daily pipeline)
├── src/
│   ├── train.py                 # Model training (XGBoost)
│   ├── drift_detector.py        # PSI + Evidently drift engine
│   ├── explainer.py             # SHAP explainability module
│   └── api.py                   # FastAPI inference server
├── monitoring/
│   ├── prometheus.yml           # Metrics scraping config
│   └── grafana_dashboard.json   # 4-panel live dashboard
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis
├── tests/
│   ├── test_drift.py            # Drift detection tests
│   └── test_api.py              # API endpoint tests
├── data/
│   ├── raw/                     # Kaggle CSV files (gitignored)
│   └── processed/               # Feature-engineered data
├── models/                      # Saved model artifacts
├── docker-compose.yml           # Full stack: API + MLflow + Airflow + Monitoring
└── requirements.txt
```

---

## 🚀 6-Component Stack

| # | Component | Tech | Purpose |
|---|-----------|------|---------|
| 1 | **Fraud Classifier** | XGBoost + SHAP | Predict fraud, explain why |
| 2 | **Inference Server** | FastAPI | Serve predictions <100ms |
| 3 | **Experiment Tracking** | MLflow | Track runs, manage model versions |
| 4 | **Drift Detection** | Evidently + PSI | Detect when data changes |
| 5 | **Orchestration** | Apache Airflow | Daily pipeline: ingest→check→retrain |
| 6 | **Monitoring** | Prometheus + Grafana | 4-panel live dashboard |

---

## 📊 Model Performance

| Metric | Score | Target |
|--------|-------|--------|
| AUC-ROC | — | > 0.88 |
| AUC-PR | — | — |
| Dataset | IEEE-CIS (590K transactions) | — |

---

## ⚡ Quick Start

### 1. Download Data
```bash
# Install Kaggle CLI
pip install kaggle

# Download IEEE-CIS dataset
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train Baseline Model
```bash
python src/train.py
```

### 5. Start Full Stack (Week 2+)
```bash
docker-compose up
```

---

## 🗓️ Build Timeline

| Week | Milestone | Status |
|------|-----------|--------|
| Week 1 | EDA + Baseline XGBoost (AUC-ROC > 0.88) | ✅ |
| Week 2 | SHAP Explainer + FastAPI + Docker | 🔄 |
| Week 3 | MLflow Tracking + Model Registry | ⏳ |
| Week 4 | Drift Detection (PSI + Evidently) | ⏳ |
| Week 5 | Airflow DAG + Prometheus + Grafana | ⏳ |
| Week 6 | CI/CD + Tests + Demo Video + Deploy | ⏳ |

---

## 🎯 Target Roles

Built for **MLOps Engineer** and **FinTech AI** positions at:
Razorpay · PhonePe · CRED · Zepto · Swiggy · AI startups

---

*Built by [Your Name] — B.Tech IT, SRM Delhi NCR, Class of 2026*
