"""
src/explainer.py
================
SHAP explainability module — Week 3: MLflow registry-aware version.

WHAT CHANGED FROM WEEK 2:
  - Can load model from MLflow registry by stage name ("Production")
  - Falls back to local .joblib if MLflow is not available
  - feature_names loaded from MLflow artifact if loaded from registry
  - All SHAP logic unchanged

LOAD PRIORITY:
  1. MLflow registry stage "Production" (production default)
  2. MLflow registry stage "Staging"   (fallback if no Production)
  3. Local models/fraud_model_v1.joblib (fallback for local dev)
"""

import json
import os
import time
import numpy as np
import pandas as pd
import joblib
import shap


# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────

MODEL_PATH          = "models/fraud_model_v1.joblib"
FEATURE_NAMES_PATH  = "models/feature_names.json"
TOP_N_REASONS       = 5
MISSING_SENTINEL    = -999.0

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
REGISTERED_MODEL_NAME = "fraud-detector"

FEATURE_LABELS = {
    "Transaction_hour":        "Hour of transaction",
    "Transaction_day":         "Day of week",
    "TransactionAmt":          "Transaction amount ($)",
    "TransactionAmt_log":      "Transaction amount (log-scaled)",
    "TransactionAmt_rounded":  "Amount is a round number",
    "card1":                   "Card identifier",
    "card2":                   "Card bin number",
    "card3":                   "Card country code",
    "card4":                   "Card network (Visa/MC/etc)",
    "card5":                   "Card product type",
    "card6":                   "Debit vs credit",
    "addr1":                   "Billing zip code",
    "addr2":                   "Billing country",
    "dist1":                   "Distance: transaction to billing",
    "dist2":                   "Distance: transaction to shipping",
    "P_emaildomain":           "Purchaser email domain",
    "R_emaildomain":           "Recipient email domain",
    "C1":                      "Count: cards linked to address",
    "C2":                      "Count: cards linked to email",
    "C6":                      "Count: days since last transaction",
    "C13":                     "Count: unique billing addresses",
    "D1":                      "Days since last transaction",
    "D4":                      "Days since first transaction on card",
    "D10":                     "Days since device last seen",
    "D15":                     "Days since last address match",
}

RISK_LEVELS = [
    (0.80, "CRITICAL",  "Automatically decline. Extremely high fraud probability."),
    (0.60, "HIGH",      "Manual review required before processing."),
    (0.40, "MEDIUM",    "Flag for monitoring. Allow with extra auth (OTP/CVV)."),
    (0.20, "LOW",       "Some risk signals present. Monitor transaction."),
    (0.00, "MINIMAL",   "Transaction looks legitimate. Process normally."),
]


# ──────────────────────────────────────────────────────────
# MODEL LOADING — MLflow Registry or Local Fallback
# ──────────────────────────────────────────────────────────

def load_model_from_registry(stage: str = "Production"):
    """
    Load model from MLflow model registry by stage name.

    WHY LOAD BY STAGE, NOT BY VERSION NUMBER?
    If you hardcode version=3, you need a code change to deploy v4.
    Loading by stage="Production" means: "give me whatever is currently
    in Production". When you promote v4, the API automatically serves it.
    No code change, no redeploy. This is the registry's main value.

    URI FORMAT: models:/fraud-detector/Production
      models:/           → MLflow model registry scheme
      fraud-detector     → registered model name
      Production         → stage (or a version number like /3)
    """
    import mlflow.xgboost
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Check if the requested stage has a model
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[stage])
    if not versions:
        raise ValueError(f"No model in stage '{stage}' for '{REGISTERED_MODEL_NAME}'")

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{stage}"
    print(f"  📦 Loading from registry: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    # Load feature names from the run's artifacts
    latest = versions[0]
    run_id = latest.run_id
    artifact_uri = client.download_artifacts(
        run_id, "model_assets/feature_names.json"
    )
    with open(artifact_uri) as f:
        feature_names = json.load(f)

    version_info = {
        "source":  "mlflow_registry",
        "stage":   stage,
        "version": latest.version,
        "run_id":  run_id,
    }
    return model, feature_names, version_info


def load_model_local():
    """Fallback: load from local .joblib file."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
    version_info = {"source": "local_file", "path": MODEL_PATH}
    return model, feature_names, version_info


def load_best_available_model():
    """
    Try loading from MLflow registry first, fall back to local file.

    LOAD ORDER:
      1. MLflow "Production" stage  → what the API should always use
      2. MLflow "Staging" stage     → if no Production model yet
      3. Local .joblib file         → for dev without MLflow server

    This means:
    - First run (no registry yet): uses local .joblib from train.py
    - After first mlflow-integrated train.py run: uses registry
    - MLflow server down: gracefully falls back to local file
    """
    # Try Production first
    try:
        model, feature_names, info = load_model_from_registry("Production")
        print(f"  ✅ Loaded from MLflow registry (Production v{info['version']})")
        return model, feature_names, info
    except Exception as e:
        print(f"  ⚠️  MLflow Production not available: {e}")

    # Try Staging
    try:
        model, feature_names, info = load_model_from_registry("Staging")
        print(f"  ✅ Loaded from MLflow registry (Staging v{info['version']})")
        return model, feature_names, info
    except Exception as e:
        print(f"  ⚠️  MLflow Staging not available: {e}")

    # Fall back to local
    print("  ⚠️  Falling back to local model file")
    model, feature_names, info = load_model_local()
    print(f"  ✅ Loaded from local file: {info['path']}")
    return model, feature_names, info


# ──────────────────────────────────────────────────────────
# EXPLAINER CLASS
# ──────────────────────────────────────────────────────────

class FraudExplainer:

    def __init__(self, model_path: str = MODEL_PATH,
                 feature_names_path: str = FEATURE_NAMES_PATH,
                 use_registry: bool = True):
        """
        Load model + feature names + build SHAP explainer.

        Args:
            model_path: fallback path if MLflow not available
            feature_names_path: fallback path if MLflow not available
            use_registry: if True, try MLflow registry first (default)
        """
        print("🔍 Loading FraudExplainer...")

        env_registry = os.getenv("USE_REGISTRY", "true").lower() != "false"
        if use_registry and env_registry:
            self.model, self.feature_names, self.version_info = \
                load_best_available_model()
        else:
            self.model, self.feature_names, self.version_info = \
                load_model_local()

        print(f"  ✅ Model loaded ({len(self.feature_names)} features)")
        print(f"  ✅ Source: {self.version_info['source']}")

        self.explainer = shap.TreeExplainer(self.model)
        print("  ✅ SHAP TreeExplainer built")

        # Warm-up call — eliminates first-request latency spike
        print("  🔥 Running warm-up prediction...")
        t0 = time.perf_counter()
        dummy = pd.DataFrame(
            [[MISSING_SENTINEL] * len(self.feature_names)],
            columns=self.feature_names
        )
        _ = self.explainer.shap_values(dummy)
        print(f"  ✅ Warm-up done in {(time.perf_counter()-t0)*1000:.0f}ms")
        print("  ✅ FraudExplainer ready\n")

    def get_model_version(self) -> str:
        """Return a human-readable model version string for API responses."""
        info = self.version_info
        if info["source"] == "mlflow_registry":
            return f"v{info['version']}-{info['stage']}"
        return "v1.0.0-local"

    def _get_risk_level(self, score: float) -> tuple[str, str]:
        for threshold, level, action in RISK_LEVELS:
            if score >= threshold:
                return level, action
        return "MINIMAL", "Transaction looks legitimate."

    def _get_feature_label(self, feature_name: str) -> str:
        if feature_name in FEATURE_LABELS:
            return FEATURE_LABELS[feature_name]
        return feature_name.replace("_", " ")

    def _build_reason_text(self, feature_name: str, shap_value: float,
                           feature_value: float) -> str:
        label = self._get_feature_label(feature_name)
        direction = "increases fraud risk" if shap_value > 0 else "decreases fraud risk"

        if feature_name == "TransactionAmt":
            val_str = f"${feature_value:,.2f}"
        elif feature_name == "Transaction_hour":
            val_str = f"{int(feature_value):02d}:00"
        elif feature_name == "Transaction_day":
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            val_str = days[int(feature_value) % 7]
        elif abs(feature_value) > 1000:
            val_str = f"{feature_value:,.0f}"
        elif feature_value == int(feature_value):
            val_str = str(int(feature_value))
        else:
            val_str = f"{feature_value:.3f}"

        return f"{label} ({val_str}) {direction} ({shap_value:+.3f})"

    def prepare_input(self, transaction: dict) -> pd.DataFrame:
        row = {feat: MISSING_SENTINEL for feat in self.feature_names}
        for key, val in transaction.items():
            if key in row:
                row[key] = val
        if "TransactionDT" in transaction:
            dt = transaction["TransactionDT"]
            row["Transaction_hour"] = (dt // 3600) % 24
            row["Transaction_day"]  = (dt // (3600 * 24)) % 7
        if "TransactionAmt" in transaction:
            amt = transaction["TransactionAmt"]
            row["TransactionAmt_log"]     = np.log1p(amt)
            row["TransactionAmt_rounded"] = float(amt % 1 == 0)
        return pd.DataFrame([row])[self.feature_names]

    def explain_prediction(self, transaction: dict) -> dict:
        X = self.prepare_input(transaction)
        feature_values = X.iloc[0].to_dict()

        fraud_score = float(self.model.predict_proba(X)[0, 1])
        risk_level, risk_action = self._get_risk_level(fraud_score)

        shap_vals = self.explainer.shap_values(X)
        shap_array = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

        shap_dict = {
            feat: float(val)
            for feat, val in zip(self.feature_names, shap_array)
        }

        meaningful     = {}
        missing_signals = {}
        for feat, shap_val in shap_dict.items():
            if feature_values.get(feat, MISSING_SENTINEL) == MISSING_SENTINEL:
                missing_signals[feat] = shap_val
            else:
                meaningful[feat] = shap_val

        top_meaningful = sorted(
            meaningful.items(), key=lambda x: abs(x[1]), reverse=True
        )[:TOP_N_REASONS]

        top_missing = sorted(
            missing_signals.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        top_reasons = [
            self._build_reason_text(feat, val, feature_values[feat])
            for feat, val in top_meaningful
        ]

        if len(top_reasons) < 3:
            for feat, val in top_missing[:3 - len(top_reasons)]:
                label = self._get_feature_label(feat)
                direction = "increases fraud risk" if val > 0 else "decreases fraud risk"
                top_reasons.append(
                    f"{label} (not provided) {direction} ({val:+.3f})"
                )

        base_score = float(self.explainer.expected_value)
        if isinstance(base_score, (list, np.ndarray)):
            base_score = float(base_score[1])

        return {
            "fraud_score":    round(fraud_score, 4),
            "risk_level":     risk_level,
            "risk_action":    risk_action,
            "top_reasons":    top_reasons,
            "shap_values":    {k: round(v, 4) for k, v in top_meaningful},
            "missing_signals":{k: round(v, 4) for k, v in top_missing},
            "base_score":     round(base_score, 4),
            "feature_values": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in feature_values.items()
                if v != MISSING_SENTINEL
            },
        }


# ──────────────────────────────────────────────────────────
# STANDALONE TEST — python src/explainer.py
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    explainer = FraudExplainer()

    tx = {
        "TransactionAmt": 500.00,
        "TransactionDT":  10800,
        "card4": 1, "card6": 0,
        "D1": 0, "D10": 0, "dist1": 500,
    }

    t0 = time.perf_counter()
    result = explainer.explain_prediction(tx)
    ms = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print(f"  Score:   {result['fraud_score']} ({result['risk_level']})")
    print(f"  Latency: {ms:.1f}ms")
    print(f"  Source:  {explainer.version_info}")
    print(f"\n  Top reasons:")
    for i, r in enumerate(result["top_reasons"], 1):
        print(f"    {i}. {r}")