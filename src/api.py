"""
src/api.py
==========
FastAPI inference server for the fraud detection pipeline.

WHAT IS FASTAPI? (Simple English)
-----------------------------------
FastAPI is a web framework — it lets your Python code respond to HTTP requests.
Think of it like a waiter:
  - Customer (your app/bank) sends an order (POST /predict with transaction JSON)
  - Waiter (FastAPI) takes the order to the kitchen (fraud model)
  - Kitchen returns the food (fraud score + explanation)
  - Waiter brings it back (JSON response)

WHY FASTAPI AND NOT FLASK?
  - FastAPI auto-generates API docs at /docs (huge for demos)
  - It validates request/response types automatically (Pydantic)
  - It's 2-3x faster than Flask for ML workloads
  - It's what Razorpay/PhonePe/Zepto teams actually use

ENDPOINTS:
  POST /predict         → score a transaction, get SHAP explanation
  GET  /drift-report    → current PSI scores per feature (Week 4)
  GET  /health          → liveness check for Docker/Kubernetes
  GET  /metrics         → Prometheus metrics scraping
  GET  /docs            → auto-generated Swagger UI (free!)

PERFORMANCE TARGET: < 100ms per /predict call
HOW WE HIT IT:
  - Model loaded ONCE at startup (not per request)
  - SHAP TreeExplainer is O(n_trees * depth), not O(n_samples)
  - FastAPI runs async so concurrent requests don't block each other
"""

import time
import json
import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi.testclient import TestClient

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np

# Prometheus client — exposes /metrics for Grafana to scrape
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

# Our modules
from src.explainer import FraudExplainer


# ──────────────────────────────────────────────────────────
# PROMETHEUS METRICS DEFINITIONS
# ──────────────────────────────────────────────────────────
# These are the 4 things Grafana will display on its dashboard.
# Every time our code runs, we .inc() or .observe() these.

# Total predictions made, labelled by risk level
PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions made",
    ["risk_level"]   # label — Grafana can filter by CRITICAL/HIGH/MEDIUM/etc
)

# Histogram of fraud scores (0.0 to 1.0 in buckets)
FRAUD_SCORE_HISTOGRAM = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# API latency in seconds
LATENCY_HISTOGRAM = Histogram(
    "predict_latency_seconds",
    "Time taken per /predict call",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
    # The 0.1 bucket = 100ms — our SLA target
)

# Live gauge: current PSI score for the most drifted feature
DRIFT_GAUGE = Gauge(
    "feature_drift_psi_max",
    "Maximum PSI score across all monitored features (0.2+ = retrain)"
)

# Retraining event counter
RETRAIN_COUNTER = Counter(
    "model_retraining_events_total",
    "Number of times the model was automatically retrained"
)


# ──────────────────────────────────────────────────────────
# GLOBAL STATE
# Model is loaded ONCE when the server starts, not per-request
# This is the "warm model" pattern — critical for latency
# ──────────────────────────────────────────────────────────

_explainer: Optional[FraudExplainer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context — runs startup code before serving requests.
    WHAT IS THIS? It's FastAPI's way of saying "run this before you start
    accepting traffic". We use it to load the model.
    """
    global _explainer
    print("\n🚀 API starting up...")

    model_path = os.getenv("MODEL_PATH", "models/fraud_model_v1.joblib")
    feature_path = os.getenv("FEATURE_NAMES_PATH", "models/feature_names.json")

    try:
        _explainer = FraudExplainer(
            model_path=model_path,
            feature_names_path=feature_path
        )
        print("✅ Model loaded and ready to serve\n")
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        print("   API will start but /predict will return 503 until model is loaded.\n")

    yield  # server is live and serving requests here

    # Shutdown code (cleanup) goes after yield
    print("\n⏹️  API shutting down...")


# ──────────────────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="""
    ## Self-Healing MLOps Fraud Detection Pipeline

    Scores financial transactions for fraud probability using XGBoost + SHAP explainability.

    ### How it works
    1. Send a transaction via `POST /predict`
    2. Get back a fraud score (0-1), risk level, and human-readable reasons
    3. Monitor drift via `GET /drift-report`
    4. Prometheus scrapes `GET /metrics` for Grafana dashboards

    **Performance:** < 100ms p99 latency on CPU
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# 🔥 FIX FOR PYTEST — force lifespan to run
if os.getenv("PYTEST_RUNNING") == "1":
    import asyncio
    asyncio.run(lifespan(app).__aenter__())
    
# CORS — allows the API to be called from browsers/frontend apps
# In production, replace "*" with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# WHAT IS PYDANTIC? It's like a type-checker for your API.
# If someone sends a string where you expect a float, Pydantic
# rejects it with a clear error BEFORE it reaches your model.
# ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Input schema for POST /predict.
    Only the most important fields are required.
    Everything else is optional (missing features → -999 sentinel).
    """
    # Required core fields
    TransactionAmt: float = Field(
        ...,
        gt=0,
        description="Transaction amount in USD. Must be positive.",
        example=149.99
    )

    # Optional fields — map directly to IEEE-CIS features
    TransactionDT: Optional[int] = Field(
        None,
        description="Seconds since reference point (used to extract hour/day)",
        example=86400
    )
    ProductCD: Optional[int] = Field(None, example=0)
    card1: Optional[float] = Field(None, example=12345)
    card2: Optional[float] = Field(None, example=325)
    card3: Optional[float] = Field(None, example=150)
    card4: Optional[int]   = Field(None, example=1, description="Card network (encoded)")
    card5: Optional[float] = Field(None, example=226)
    card6: Optional[int]   = Field(None, example=0, description="Debit=0, Credit=1")
    addr1: Optional[float] = Field(None, example=315)
    addr2: Optional[float] = Field(None, example=87)
    dist1: Optional[float] = Field(None, example=19.0)
    dist2: Optional[float] = Field(None, example=None)
    P_emaildomain: Optional[int] = Field(None, example=3)
    R_emaildomain: Optional[int] = Field(None, example=3)
    C1:  Optional[float] = Field(None, example=1)
    C2:  Optional[float] = Field(None, example=1)
    C6:  Optional[float] = Field(None, example=1)
    C13: Optional[float] = Field(None, example=28)
    D1:  Optional[float] = Field(None, example=0, description="Days since last txn")
    D4:  Optional[float] = Field(None, example=0)
    D10: Optional[float] = Field(None, example=0, description="Days since device seen")
    D15: Optional[float] = Field(None, example=0)

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_reasonable(cls, v):
        if v > 1_000_000:
            raise ValueError("TransactionAmt exceeds $1,000,000 — likely a data error")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "TransactionAmt": 500.00,
            "TransactionDT": 10800,
            "card4": 1,
            "card6": 0,
            "D1": 0,
            "D10": 0,
            "dist1": 500,
        }
    }}


class PredictionResponse(BaseModel):
    """Output schema for POST /predict."""
    fraud_score:    float  = Field(..., ge=0, le=1, description="Fraud probability (0=legit, 1=fraud)")
    risk_level:     str    = Field(..., description="MINIMAL/LOW/MEDIUM/HIGH/CRITICAL")
    risk_action:    str    = Field(..., description="Recommended action")
    top_reasons:    list[str] = Field(..., description="Human-readable SHAP explanations")
    shap_values:    dict   = Field(..., description="Raw SHAP values per feature")
    base_score:     float  = Field(..., description="Average fraud rate (model baseline)")
    latency_ms:     float  = Field(..., description="Server-side prediction latency in ms")
    model_version:  str    = Field(..., description="Model version used for this prediction")


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    uptime_seconds: float


class DriftReport(BaseModel):
    """Output schema for GET /drift-report."""
    status:          str   = Field(..., description="stable/warning/drifted")
    max_psi:         float = Field(..., description="Highest PSI across all features")
    drift_threshold: float = Field(0.2, description="PSI threshold for retraining")
    features:        dict  = Field(..., description="PSI score per monitored feature")
    recommendation:  str   = Field(..., description="Human-readable drift recommendation")
    last_checked:    str   = Field(..., description="When drift was last computed")


# ──────────────────────────────────────────────────────────
# SERVER START TIME (for uptime tracking)
# ──────────────────────────────────────────────────────────

_server_start_time = time.time()
_model_version = "v1.0.0"


# ──────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """
    Liveness probe — Docker and Kubernetes call this to check if the
    container is healthy. If this returns non-200, the container gets restarted.
    """
    return HealthResponse(
        status="healthy" if _explainer is not None else "degraded",
        model_loaded=_explainer is not None,
        model_version=_model_version,
        uptime_seconds=round(time.time() - _server_start_time, 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(transaction: TransactionRequest, request: Request):
    """
    Score a single transaction for fraud.

    - Runs XGBoost prediction (probability)
    - Computes SHAP values (explainability)
    - Returns score + human-readable reasons + raw SHAP

    **Latency target:** < 100ms p99
    """
    if _explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py first, then restart the API."
        )

    # ── Time the prediction ──────────────────────
    t_start = time.perf_counter()

    # Convert Pydantic model → plain dict (exclude unset fields)
    tx_dict = transaction.model_dump(exclude_none=True)

    # Run SHAP explanation
    result = _explainer.explain_prediction(tx_dict)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── Update Prometheus metrics ────────────────
    PREDICTION_COUNTER.labels(risk_level=result["risk_level"]).inc()
    FRAUD_SCORE_HISTOGRAM.observe(result["fraud_score"])
    LATENCY_HISTOGRAM.observe(latency_ms / 1000)  # histogram expects seconds

    # ── Log slow predictions (> 100ms) ──────────
    if latency_ms > 100:
        print(f"⚠️  Slow prediction: {latency_ms:.1f}ms  (target: <100ms)")

    return PredictionResponse(
        fraud_score=result["fraud_score"],
        risk_level=result["risk_level"],
        risk_action=result["risk_action"],
        top_reasons=result["top_reasons"],
        shap_values=result["shap_values"],
        base_score=result["base_score"],
        latency_ms=round(latency_ms, 2),
        model_version=_model_version,
    )


@app.get("/drift-report", response_model=DriftReport, tags=["Monitoring"])
async def drift_report():
    """
    Return current PSI (Population Stability Index) scores per feature.

    PSI < 0.1  → stable
    PSI 0.1-0.2 → warning
    PSI > 0.2  → retrain triggered

    Full implementation in Week 4. This returns a demo response for now.
    """
    import datetime

    # Week 4 will replace this with real PSI computation from drift_detector.py
    # For now, return a realistic demo payload so the /docs page shows correctly
    demo_features = {
        "TransactionAmt":   0.04,   # stable
        "Transaction_hour": 0.07,   # stable
        "card1":            0.12,   # warning
        "dist1":            0.18,   # warning (close to threshold!)
        "C1":               0.03,   # stable
        "D1":               0.09,   # stable
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

    # Update Prometheus gauge
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
    """
    Prometheus metrics endpoint.
    Grafana scrapes this every 15 seconds to build dashboard panels.

    Metrics exposed:
    - fraud_predictions_total (by risk_level)
    - fraud_score_distribution (histogram)
    - predict_latency_seconds (histogram, p99 is our SLA)
    - feature_drift_psi_max (gauge)
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Operations"])
async def root():
    """Root endpoint — returns API info and available endpoints."""
    return {
        "name":        "Fraud Detection API",
        "version":     _model_version,
        "status":      "running",
        "model_loaded": _explainer is not None,
        "endpoints": {
            "predict":      "POST /predict",
            "drift_report": "GET /drift-report",
            "health":       "GET /health",
            "metrics":      "GET /metrics",
            "docs":         "GET /docs",
        }
    }


# ──────────────────────────────────────────────────────────
# LOCAL DEV RUNNER
# python src/api.py  →  starts server at http://localhost:8000
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,     # auto-restart on file changes (dev only)
        log_level="info",
    )
