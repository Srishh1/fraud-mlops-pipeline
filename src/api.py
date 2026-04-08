"""
src/api.py
==========
FastAPI inference server — Week 3: MLflow registry-aware version.

WHAT CHANGED FROM WEEK 2:
  - model_version in responses now reflects MLflow registry version
    e.g. "v3-Production" instead of hardcoded "v1.0.0"
  - /health endpoint reports model source (registry vs local)
  - /model-info endpoint added: shows current registry metadata
  - FraudExplainer init passes use_registry=True (tries MLflow first)
  - Everything else unchanged
"""

import time
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

from src.explainer import FraudExplainer


# ──────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ──────────────────────────────────────────────────────────

PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions made",
    ["risk_level"]
)
FRAUD_SCORE_HISTOGRAM = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
LATENCY_HISTOGRAM = Histogram(
    "predict_latency_seconds",
    "Time taken per /predict call",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)
DRIFT_GAUGE = Gauge(
    "feature_drift_psi_max",
    "Maximum PSI score across all monitored features"
)
RETRAIN_COUNTER = Counter(
    "model_retraining_events_total",
    "Number of times the model was automatically retrained"
)


# ──────────────────────────────────────────────────────────
# GLOBAL STATE
# ──────────────────────────────────────────────────────────

_explainer: Optional[FraudExplainer] = None
_server_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _explainer
    print("\n🚀 API starting up...")

    try:
        # use_registry=True → tries MLflow Production, then Staging, then local
        _explainer = FraudExplainer(use_registry=True)
        print("✅ Model loaded and ready to serve\n")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("   API will start but /predict returns 503 until model loads.\n")

    yield
    print("\n⏹️  API shutting down...")


# ──────────────────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="""
## Self-Healing MLOps Fraud Detection Pipeline

Scores financial transactions for fraud probability using XGBoost + SHAP.
Model loaded from **MLflow Model Registry** (stage: Production).

### Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Score a transaction |
| `/drift-report` | GET | PSI drift scores per feature |
| `/model-info` | GET | Current model registry metadata |
| `/health` | GET | Liveness probe |
| `/metrics` | GET | Prometheus scrape target |
| `/docs` | GET | This page |

**Latency target:** < 100ms p99
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# SCHEMAS
# ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0, example=149.99)
    TransactionDT:  Optional[int]   = Field(None, example=86400)
    ProductCD:      Optional[int]   = Field(None, example=0)
    card1:          Optional[float] = Field(None, example=12345)
    card2:          Optional[float] = Field(None, example=325)
    card3:          Optional[float] = Field(None, example=150)
    card4:          Optional[int]   = Field(None, example=1)
    card5:          Optional[float] = Field(None, example=226)
    card6:          Optional[int]   = Field(None, example=0)
    addr1:          Optional[float] = Field(None, example=315)
    addr2:          Optional[float] = Field(None, example=87)
    dist1:          Optional[float] = Field(None, example=19.0)
    dist2:          Optional[float] = Field(None, example=None)
    P_emaildomain:  Optional[int]   = Field(None, example=3)
    R_emaildomain:  Optional[int]   = Field(None, example=3)
    C1:             Optional[float] = Field(None, example=1)
    C2:             Optional[float] = Field(None, example=1)
    C6:             Optional[float] = Field(None, example=1)
    C13:            Optional[float] = Field(None, example=28)
    D1:             Optional[float] = Field(None, example=0)
    D4:             Optional[float] = Field(None, example=0)
    D10:            Optional[float] = Field(None, example=0)
    D15:            Optional[float] = Field(None, example=0)

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_reasonable(cls, v):
        if v > 1_000_000:
            raise ValueError("TransactionAmt exceeds $1,000,000 — likely a data error")
        return v

    model_config = {"json_schema_extra": {"example": {
        "TransactionAmt": 500.00,
        "TransactionDT": 10800,
        "card4": 1, "card6": 0,
        "D1": 0, "D10": 0, "dist1": 500,
    }}}


class PredictionResponse(BaseModel):
    fraud_score:     float     = Field(..., ge=0, le=1)
    risk_level:      str       = Field(...)
    risk_action:     str       = Field(...)
    top_reasons:     list[str] = Field(...)
    shap_values:     dict      = Field(...)
    missing_signals: dict      = Field(...)
    base_score:      float     = Field(...)
    latency_ms:      float     = Field(...)
    model_version:   str       = Field(...)


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    model_version:  str
    model_source:   str    # "mlflow_registry" or "local_file"
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """NEW in Week 3 — exposes MLflow registry metadata."""
    model_name:    str
    version:       str
    stage:         str
    source:        str
    run_id:        Optional[str]
    tracking_uri:  str


class DriftReport(BaseModel):
    status:          str
    max_psi:         float
    drift_threshold: float
    features:        dict
    recommendation:  str
    last_checked:    str


# ──────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    model_source = "none"
    model_version = "none"
    if _explainer:
        model_source  = _explainer.version_info.get("source", "unknown")
        model_version = _explainer.get_model_version()

    return HealthResponse(
        status="healthy" if _explainer is not None else "degraded",
        model_loaded=_explainer is not None,
        model_version=model_version,
        model_source=model_source,
        uptime_seconds=round(time.time() - _server_start_time, 1),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Operations"])
async def model_info():
    """
    NEW in Week 3 — returns current model registry metadata.

    WHAT THIS IS USEFUL FOR:
    - Auditing: which exact model version is serving traffic right now?
    - Debugging: did the promotion go through? is the API using the new model?
    - Dashboards: Grafana can scrape this to show "current model version"

    In production fintech, this is part of the model governance audit trail.
    RBI regulations in India require knowing exactly which model made each decision.
    """
    if _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = _explainer.version_info
    return ModelInfoResponse(
        model_name=info.get("source", "unknown"),
        version=info.get("version", "local"),
        stage=info.get("stage", "local"),
        source=info.get("source", "local_file"),
        run_id=info.get("run_id"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "mlruns"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(transaction: TransactionRequest):
    if _explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py first, then restart the API."
        )

    t_start = time.perf_counter()
    tx_dict = transaction.model_dump(exclude_none=True)
    result  = _explainer.explain_prediction(tx_dict)
    latency_ms = (time.perf_counter() - t_start) * 1000

    PREDICTION_COUNTER.labels(risk_level=result["risk_level"]).inc()
    FRAUD_SCORE_HISTOGRAM.observe(result["fraud_score"])
    LATENCY_HISTOGRAM.observe(latency_ms / 1000)

    if latency_ms > 100:
        print(f"⚠️  Slow prediction: {latency_ms:.1f}ms  (target: <100ms)")

    return PredictionResponse(
        fraud_score=result["fraud_score"],
        risk_level=result["risk_level"],
        risk_action=result["risk_action"],
        top_reasons=result["top_reasons"],
        shap_values=result["shap_values"],
        missing_signals=result["missing_signals"],
        base_score=result["base_score"],
        latency_ms=round(latency_ms, 2),
        model_version=_explainer.get_model_version(),
    )


@app.get("/drift-report", response_model=DriftReport, tags=["Monitoring"])
async def drift_report():
    """PSI drift scores per feature. Full implementation in Week 4."""
    import datetime

    demo_features = {
        "TransactionAmt":   0.04,
        "Transaction_hour": 0.07,
        "card1":            0.12,
        "dist1":            0.18,
        "C1":               0.03,
        "D1":               0.09,
    }
    max_psi = max(demo_features.values())

    if max_psi >= 0.2:
        status = "drifted"
        recommendation = "Significant drift detected. Airflow DAG will trigger retraining."
    elif max_psi >= 0.1:
        status = "warning"
        recommendation = "Moderate drift detected. Monitor closely over next 24h."
    else:
        status = "stable"
        recommendation = "All features stable. No action required."

    DRIFT_GAUGE.set(max_psi)

    return DriftReport(
        status=status,
        max_psi=round(max_psi, 4),
        drift_threshold=0.2,
        features=demo_features,
        recommendation=recommendation,
        last_checked=datetime.datetime.utcnow().isoformat() + "Z",
    )


@app.get("/metrics", tags=["Operations"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Operations"])
async def root():
    model_version = _explainer.get_model_version() if _explainer else "not loaded"
    return {
        "name":          "Fraud Detection API",
        "version":       model_version,
        "status":        "running",
        "model_loaded":  _explainer is not None,
        "endpoints": {
            "predict":     "POST /predict",
            "drift_report":"GET /drift-report",
            "model_info":  "GET /model-info",
            "health":      "GET /health",
            "metrics":     "GET /metrics",
            "docs":        "GET /docs",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)