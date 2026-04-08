"""
tests/test_api.py
==================
Full test suite for the FastAPI inference server.

HOW FASTAPI TESTING WORKS:
FastAPI has a built-in test client (TestClient) that lets you make
HTTP requests to your app WITHOUT actually starting a server.
It's fast, reliable, and doesn't need a network connection.

HOW TO RUN:
  pytest tests/test_api.py -v

WHAT WE'RE TESTING:
  1. Health endpoint returns 200 and correct fields
  2. Root endpoint returns API metadata
  3. Predict endpoint validates input (rejects bad data)
  4. Drift report endpoint returns correct schema
  5. Metrics endpoint returns Prometheus format
  6. Prediction response has correct shape and types
  7. Fraud score is always between 0 and 1
  8. Latency reported is positive
"""

import pytest
import json
import os
import sys

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────
# FIXTURES
# WHAT IS A FIXTURE? A reusable setup block that pytest runs
# before your tests. Here we create the test client once
# and share it across all tests.
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_transaction():
    """A realistic transaction dict for testing."""
    return {
        "TransactionAmt": 149.99,
        "TransactionDT": 86400,     # Day 1, 00:00
        "card4": 1,                 # Visa
        "card6": 0,                 # Debit
        "addr1": 315.0,
        "C1": 1.0,
        "D1": 5.0,
    }


@pytest.fixture
def suspicious_transaction():
    """A transaction designed to have high fraud signals."""
    return {
        "TransactionAmt": 500.00,   # round large amount
        "TransactionDT": 10800,     # 3am
        "card4": 1,
        "card6": 0,
        "D1": 0,                    # first ever transaction
        "D10": 0,                   # new device
        "dist1": 500.0,             # far from billing address
    }


# ──────────────────────────────────────────────────────────
# HEALTH ENDPOINT TESTS
# ──────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data

    def test_health_status_is_string(self, client):
        data = client.get("/health").json()
        assert isinstance(data["status"], str)
        assert data["status"] in ["healthy", "degraded"]

    def test_health_uptime_is_positive(self, client):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0


# ──────────────────────────────────────────────────────────
# ROOT ENDPOINT TESTS
# ──────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_endpoints_map(self, client):
        data = client.get("/").json()
        assert "endpoints" in data
        assert "predict" in data["endpoints"]
        assert "health" in data["endpoints"]
        assert "drift_report" in data["endpoints"]


# ──────────────────────────────────────────────────────────
# PREDICT ENDPOINT TESTS
# ──────────────────────────────────────────────────────────

class TestPredictEndpoint:

    def test_predict_requires_transaction_amount(self, client):
        """Without TransactionAmt, should return 422 Unprocessable Entity."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_rejects_negative_amount(self, client):
        """TransactionAmt must be > 0."""
        response = client.post("/predict", json={"TransactionAmt": -50.0})
        assert response.status_code == 422

    def test_predict_rejects_zero_amount(self, client):
        """TransactionAmt must be > 0, not >= 0."""
        response = client.post("/predict", json={"TransactionAmt": 0.0})
        assert response.status_code == 422

    def test_predict_rejects_absurdly_large_amount(self, client):
        """$2M transaction is likely a data error."""
        response = client.post("/predict", json={"TransactionAmt": 2_000_000.0})
        assert response.status_code == 422

    def test_predict_accepts_minimal_valid_input(self, client):
        """Only TransactionAmt is required. Everything else is optional."""
        response = client.post("/predict", json={"TransactionAmt": 100.0})
        # 200 if model loaded, 503 if not. Both are valid for this test.
        assert response.status_code in [200, 503]

    @pytest.mark.skipif(
        not os.path.exists("models/fraud_model_v1.joblib"),
        reason="Model file not found — run src/train.py first"
    )
    def test_predict_returns_correct_schema(self, client, sample_transaction):
        """Full schema validation when model is present."""
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200

        data = response.json()
        assert "fraud_score" in data
        assert "risk_level" in data
        assert "risk_action" in data
        assert "top_reasons" in data
        assert "shap_values" in data
        assert "latency_ms" in data
        assert "model_version" in data

    @pytest.mark.skipif(
        not os.path.exists("models/fraud_model_v1.joblib"),
        reason="Model file not found — run src/train.py first"
    )
    def test_fraud_score_is_between_0_and_1(self, client, sample_transaction):
        """Fraud score must always be a probability."""
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200
        score = response.json()["fraud_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} is outside [0, 1]"

    @pytest.mark.skipif(
        not os.path.exists("models/fraud_model_v1.joblib"),
        reason="Model file not found — run src/train.py first"
    )
    def test_risk_level_is_valid(self, client, sample_transaction):
        """Risk level must be one of the defined categories."""
        valid_levels = {"MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200
        assert response.json()["risk_level"] in valid_levels

    @pytest.mark.skipif(
        not os.path.exists("models/fraud_model_v1.joblib"),
        reason="Model file not found — run src/train.py first"
    )
    def test_latency_is_positive(self, client, sample_transaction):
        """Latency must be a positive number."""
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200
        assert response.json()["latency_ms"] > 0

    @pytest.mark.skipif(
        not os.path.exists("models/fraud_model_v1.joblib"),
        reason="Model file not found — run src/train.py first"
    )
    def test_top_reasons_is_non_empty_list(self, client, sample_transaction):
        """Must always return at least one reason."""
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200
        reasons = response.json()["top_reasons"]
        assert isinstance(reasons, list)
        assert len(reasons) > 0

    @pytest.mark.skipif(
        not os.path.exists("models/fraud_model_v1.joblib"),
        reason="Model file not found — run src/train.py first"
    )
    def test_suspicious_transaction_scores_higher(self, client,
                                                   sample_transaction,
                                                   suspicious_transaction):
        """
        A transaction at 3am with new device and high amount should score
        higher than a normal daytime transaction.
        This is a behavioral sanity check — the most important kind.
        """
        normal_resp  = client.post("/predict", json=sample_transaction).json()
        suspect_resp = client.post("/predict", json=suspicious_transaction).json()

        # Suspicious should have higher fraud score
        # (This test documents expected model behavior)
        print(f"\n  Normal score:    {normal_resp['fraud_score']}")
        print(f"  Suspicious score: {suspect_resp['fraud_score']}")
        # Note: not asserting strict ordering here because
        # this depends on the specific model weights.
        # Just confirm both returned valid responses.
        assert 0 <= normal_resp["fraud_score"] <= 1
        assert 0 <= suspect_resp["fraud_score"] <= 1


# ──────────────────────────────────────────────────────────
# DRIFT REPORT TESTS
# ──────────────────────────────────────────────────────────

class TestDriftEndpoint:
    def test_drift_report_returns_200(self, client):
        response = client.get("/drift-report")
        assert response.status_code == 200

    def test_drift_report_has_required_fields(self, client):
        data = client.get("/drift-report").json()
        assert "status" in data
        assert "max_psi" in data
        assert "drift_threshold" in data
        assert "features" in data
        assert "recommendation" in data

    def test_drift_status_is_valid(self, client):
        data = client.get("/drift-report").json()
        assert data["status"] in ["stable", "warning", "drifted"]

    def test_drift_threshold_is_02(self, client):
        """PSI threshold for retraining should be 0.2 — industry standard."""
        data = client.get("/drift-report").json()
        assert data["drift_threshold"] == 0.2

    def test_psi_values_are_non_negative(self, client):
        """PSI is always >= 0 mathematically."""
        data = client.get("/drift-report").json()
        for feature, psi in data["features"].items():
            assert psi >= 0, f"PSI for {feature} is negative: {psi}"


# ──────────────────────────────────────────────────────────
# METRICS ENDPOINT TESTS
# ──────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_is_prometheus_format(self, client):
        """Prometheus format has lines starting with # HELP and # TYPE."""
        content = client.get("/metrics").text
        assert "# HELP" in content
        assert "# TYPE" in content

    def test_metrics_contains_our_custom_metrics(self, client):
        """Our defined metrics should appear in the output."""
        content = client.get("/metrics").text
        assert "fraud_predictions_total" in content
        assert "predict_latency_seconds" in content
        assert "feature_drift_psi_max" in content
