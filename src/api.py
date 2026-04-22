"""
src/api.py
==========
FastAPI inference server — Week 4: real drift endpoint wired in.

WHAT CHANGED FROM WEEK 3:
  - /drift-report now reads from latest_drift_result.json (real PSI scores)
    instead of returning a hardcoded demo payload
  - Returns "not_run" status with helpful message if no drift check done yet
  - Everything else identical to Week 3
"""

import time
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


# ── Prometheus metrics ────────────────────────────────────────────────────────
PREDICTION_COUNTER    = Counter("fraud_predictions_total", "Total predictions", ["risk_level"])
FRAUD_SCORE_HISTOGRAM = Histogram("fraud_score_distribution", "Fraud score distribution",
                                   buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
LATENCY_HISTOGRAM     = Histogram("predict_latency_seconds", "Prediction latency",
                                   buckets=[0.01,0.025,0.05,0.075,0.1,0.25,0.5,1.0])
DRIFT_GAUGE           = Gauge("feature_drift_psi_max", "Max PSI across all features")
RETRAIN_COUNTER       = Counter("model_retraining_events_total", "Retraining events")

# ── Global state ──────────────────────────────────────────────────────────────
_explainer = None
_server_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _explainer
    print("\n API starting up...")
    
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        print("  MOCK_MODE enabled — skipping model load\n")
    else:
        try:
            # Import here, not at module level — keeps CI fast
            from src.explainer import FraudExplainer
            _explainer = FraudExplainer(use_registry=True)
            print(" Model loaded and ready\n")
        except Exception as e:
            print(f"  Model load failed: {e}\n")
    yield
    print("\n  API shutting down...")


app = FastAPI(
    title="Fraud Detection API",
    description="""
## Self-Healing MLOps Fraud Detection Pipeline

XGBoost + SHAP explainability. Model loaded from MLflow registry.

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Score a transaction |
| `/drift-report` | GET | Real PSI drift scores |
| `/model-info` | GET | Registry metadata |
| `/health` | GET | Liveness probe |
| `/metrics` | GET | Prometheus target |
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────

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
    dist2:          Optional[float] = Field(None)
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
        "TransactionAmt": 500.00, "TransactionDT": 10800,
        "card4": 1, "card6": 0, "D1": 0, "D10": 0, "dist1": 500,
    }}}


class PredictionResponse(BaseModel):
    fraud_score:     float     = Field(..., ge=0, le=1)
    risk_level:      str
    risk_action:     str
    top_reasons:     list[str]
    shap_values:     dict
    missing_signals: dict
    base_score:      float
    latency_ms:      float
    model_version:   str


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    model_version:  str
    model_source:   str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name:   str
    version:      str
    stage:        str
    source:       str
    run_id:       Optional[str]
    tracking_uri: str


class DriftReport(BaseModel):
    status:          str
    max_psi:         float
    drift_threshold: float
    features:        dict
    recommendation:  str
    last_checked:    str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    model_source  = _explainer.version_info.get("source", "unknown") if _explainer else "none"
    model_version = _explainer.get_model_version() if _explainer else "none"
    return HealthResponse(
        status="healthy" if _explainer else "degraded",
        model_loaded=_explainer is not None,
        model_version=model_version,
        model_source=model_source,
        uptime_seconds=round(time.time() - _server_start_time, 1),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Operations"])
async def model_info():
    if _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _explainer.version_info
    return ModelInfoResponse(
        model_name=REGISTERED_MODEL_NAME if "mlflow" in info.get("source","") else "fraud-detector",
        version=str(info.get("version", "local")),
        stage=info.get("stage", "local"),
        source=info.get("source", "local_file"),
        run_id=info.get("run_id"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "mlruns"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(transaction: TransactionRequest):
    if _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run src/train.py first.")

    t_start = time.perf_counter()
    result  = _explainer.explain_prediction(transaction.model_dump(exclude_none=True))
    latency_ms = (time.perf_counter() - t_start) * 1000

    PREDICTION_COUNTER.labels(risk_level=result["risk_level"]).inc()
    FRAUD_SCORE_HISTOGRAM.observe(result["fraud_score"])
    LATENCY_HISTOGRAM.observe(latency_ms / 1000)

    if latency_ms > 100:
        print(f"⚠️  Slow prediction: {latency_ms:.1f}ms")

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
    """
    Real PSI drift scores from the latest drift check.

    Reads from data/processed/drift_reports/latest_drift_result.json
    Written by:
      - python src/drift_detector.py   (manual)
      - Airflow DAG drift_check task   (daily, Week 5)
    """
    import datetime

    result_path = os.path.join("data", "processed", "drift_reports", "latest_drift_result.json")

    if os.path.exists(result_path):
        with open(result_path) as f:
            drift_result = json.load(f)
        psi_scores   = drift_result.get("psi_scores", {})
        max_psi      = drift_result.get("max_psi", 0.0)
        status       = drift_result.get("status", "unknown")
        recommendation = drift_result.get("recommendation", "")
        last_checked   = drift_result.get("checked_at", datetime.datetime.utcnow().isoformat())
    else:
        psi_scores   = {}
        max_psi      = 0.0
        status       = "not_run"
        recommendation = (
            "No drift check has been run yet. "
            "Run: python src/drift_detector.py  to generate the first report."
        )
        last_checked = datetime.datetime.utcnow().isoformat() + "Z"

    DRIFT_GAUGE.set(max_psi)

    return DriftReport(
        status=status,
        max_psi=round(max_psi, 4),
        drift_threshold=0.2,
        features=psi_scores,
        recommendation=recommendation,
        last_checked=last_checked,
    )


@app.get("/metrics", tags=["Operations"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Operations"])
async def root():
    model_version = _explainer.get_model_version() if _explainer else "not loaded"
    return {
        "name": "Fraud Detection API", "version": model_version,
        "status": "running", "model_loaded": _explainer is not None,
        "endpoints": {
            "predict": "POST /predict", "drift_report": "GET /drift-report",
            "model_info": "GET /model-info", "health": "GET /health",
            "metrics": "GET /metrics", "docs": "GET /docs",
        }
    }


# ── constant needed by model-info endpoint ────────────────────────────────────
REGISTERED_MODEL_NAME = "fraud-detector"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)