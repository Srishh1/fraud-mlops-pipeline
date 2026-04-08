"""
src/explainer.py
================
SHAP-based explainability module for the fraud detection model.

WHAT IS SHAP? (Simple English)
--------------------------------
Imagine your model is a black box — you give it a transaction and it says
"87% fraud". But WHY? Which features pushed it high?

SHAP (SHapley Additive exPlanations) opens the black box.
It assigns each feature a "contribution score":
  - Positive = pushed the fraud score UP (suspicious)
  - Negative = pushed the fraud score DOWN (looks legit)

Example output for a flagged transaction:
  {
    "fraud_score": 0.87,
    "risk_level": "HIGH",
    "top_reasons": [
      "Transaction at 3am (+0.31 impact)",
      "Amount is 8x user average (+0.24 impact)",
      "New device type seen for first time (+0.18 impact)"
    ],
    "shap_values": { "Transaction_hour": 0.31, "TransactionAmt_log": 0.24, ... }
  }

WHY DOES THIS MATTER FOR FINTECH?
-----------------------------------
- Regulators (RBI in India) require explainability for automated decisions
- Fraud ops analysts need to know WHY before blocking a transaction
- False positive investigations need reasons to give back to customers
- This is literally what Razorpay/PhonePe fraud teams use every day

WHERE SHAP IS USED IN THIS PROJECT:
- src/explainer.py    → this file, the core logic
- src/api.py          → calls explain_prediction() on every /predict request
- Week 4 drift_detector.py → SHAP values drift is itself a drift signal
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import shap


# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────

MODEL_PATH = "models/fraud_model_v1.joblib"
FEATURE_NAMES_PATH = "models/feature_names.json"

# How many top reasons to surface in the explanation
TOP_N_REASONS = 5

# Human-readable feature name mapping
# The IEEE-CIS dataset has cryptic names — we translate the important ones
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
    "C1":                      "Count: payment cards linked to address",
    "C2":                      "Count: payment cards linked to email",
    "C6":                      "Count: days since last transaction",
    "C13":                     "Count: unique billing addresses",
    "D1":                      "Days since last transaction",
    "D4":                      "Days since first transaction on this card",
    "D10":                     "Days since device last seen",
    "D15":                     "Days since last address match",
    "V258":                    "Vesta fraud score (component)",
    "V294":                    "Vesta velocity signal",
    "V307":                    "Vesta device signal",
}

# Risk thresholds — matches what most fintech companies use
RISK_LEVELS = [
    (0.80, "CRITICAL",  "Automatically decline. Extremely high fraud probability."),
    (0.60, "HIGH",      "Manual review required before processing."),
    (0.40, "MEDIUM",    "Flag for monitoring. Allow with extra auth (OTP/CVV)."),
    (0.20, "LOW",       "Some risk signals present. Monitor transaction."),
    (0.00, "MINIMAL",   "Transaction looks legitimate. Process normally."),
]


# ──────────────────────────────────────────────────────────
# EXPLAINER CLASS
# ──────────────────────────────────────────────────────────

class FraudExplainer:
    """
    Wraps the trained XGBoost model with SHAP explainability.

    WHY A CLASS AND NOT FUNCTIONS?
    Because loading the model and building the SHAP explainer is
    expensive (takes ~1-2 seconds). We do it ONCE when the API starts,
    then reuse the same object for every prediction request.
    This is the "warm model" pattern used in production ML serving.
    """

    def __init__(self, model_path: str = MODEL_PATH,
                 feature_names_path: str = FEATURE_NAMES_PATH):
        """
        Load model + feature names + build SHAP explainer.
        Called once at API startup.
        """
        print("🔍 Loading FraudExplainer...")

        # Load the trained XGBoost model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                "Run src/train.py first to generate the model."
            )
        self.model = joblib.load(model_path)
        print(f"  ✅ Model loaded from {model_path}")

        # Load feature names (so we know column order)
        with open(feature_names_path, "r") as f:
            self.feature_names = json.load(f)
        print(f"  ✅ Feature names loaded ({len(self.feature_names)} features)")

        # Build SHAP TreeExplainer
        # WHY TreeExplainer specifically?
        # SHAP has different explainers for different model types:
        #   - TreeExplainer  → for XGBoost, LightGBM, RandomForest (FAST, exact)
        #   - LinearExplainer → for linear models
        #   - DeepExplainer  → for neural networks (slow)
        # TreeExplainer runs in milliseconds, making it production-safe
        self.explainer = shap.TreeExplainer(self.model)
        print("  ✅ SHAP TreeExplainer built")
        print("  ✅ FraudExplainer ready\n")

    def _get_risk_level(self, score: float) -> tuple[str, str]:
        """Map fraud score to risk level + action description."""
        for threshold, level, action in RISK_LEVELS:
            if score >= threshold:
                return level, action
        return "MINIMAL", "Transaction looks legitimate."

    def _get_feature_label(self, feature_name: str) -> str:
        """Get human-readable label for a feature, or clean up the raw name."""
        if feature_name in FEATURE_LABELS:
            return FEATURE_LABELS[feature_name]
        # Clean up cryptic names a bit even without a mapping
        return feature_name.replace("_", " ").replace("TransactionAmt", "Amount")

    def _build_reason_text(self, feature_name: str, shap_value: float,
                           feature_value: float) -> str:
        """
        Turn a SHAP value into a human-readable sentence.

        Examples:
          "Transaction at 3am increases fraud risk (+0.31)"
          "Amount of $24.99 is within normal range (-0.12)"
        """
        label = self._get_feature_label(feature_name)
        direction = "increases fraud risk" if shap_value > 0 else "decreases fraud risk"
        impact = abs(shap_value)

        # Format the feature value nicely
        if feature_name == "TransactionAmt":
            val_str = f"${feature_value:,.2f}"
        elif feature_name == "Transaction_hour":
            val_str = f"{int(feature_value):02d}:00"
        elif feature_name == "Transaction_day":
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            val_str = days[int(feature_value) % 7]
        elif feature_value == -999:
            val_str = "missing"
        elif abs(feature_value) > 1000:
            val_str = f"{feature_value:,.0f}"
        elif feature_value == int(feature_value):
            val_str = str(int(feature_value))
        else:
            val_str = f"{feature_value:.3f}"

        return f"{label} ({val_str}) {direction} ({shap_value:+.3f})"

    def prepare_input(self, transaction: dict) -> pd.DataFrame:
        """
        Convert a raw transaction dictionary into a model-ready DataFrame.

        This is critical for production:
        - Fills missing features with -999 (same as training)
        - Enforces exact column order from training
        - Applies same feature engineering as train.py

        WHAT COULD GO WRONG HERE?
        If production data has different columns than training data,
        predictions will be wrong or crash. This function is the guard.
        """
        # Start with all features set to -999 (missing sentinel)
        row = {feat: -999.0 for feat in self.feature_names}

        # Fill in what we actually have
        for key, val in transaction.items():
            if key in row:
                row[key] = val

        # Apply feature engineering (same as train.py)
        if "TransactionDT" in transaction:
            dt = transaction["TransactionDT"]
            row["Transaction_hour"] = (dt // 3600) % 24
            row["Transaction_day"] = (dt // (3600 * 24)) % 7

        if "TransactionAmt" in transaction:
            amt = transaction["TransactionAmt"]
            row["TransactionAmt_log"] = np.log1p(amt)
            row["TransactionAmt_rounded"] = float(amt % 1 == 0)

        # Build DataFrame in exact training column order
        df = pd.DataFrame([row])[self.feature_names]
        return df

    def explain_prediction(self, transaction: dict) -> dict:
        """
        Main method: takes a raw transaction dict, returns full explanation.

        Returns:
        {
            "fraud_score": 0.87,           # probability of fraud (0-1)
            "risk_level": "HIGH",           # MINIMAL/LOW/MEDIUM/HIGH/CRITICAL
            "risk_action": "Manual review...",
            "top_reasons": [               # human-readable explanations
                "Transaction at 03:00 increases fraud risk (+0.31)",
                ...
            ],
            "shap_values": {               # raw SHAP values (for API consumers)
                "Transaction_hour": 0.31,
                "TransactionAmt_log": 0.24,
                ...
            },
            "base_score": 0.035,           # average fraud rate in training data
            "feature_values": { ... }      # actual values used for prediction
        }
        """
        # 1. Prepare input
        X = self.prepare_input(transaction)

        # 2. Get fraud probability
        fraud_score = float(self.model.predict_proba(X)[0, 1])
        risk_level, risk_action = self._get_risk_level(fraud_score)

        # 3. Compute SHAP values
        # shap_vals shape: (1, n_features) — one row, one value per feature
        shap_vals = self.explainer.shap_values(X)

        # For XGBoost binary classification, shap_values returns
        # a 2D array (n_samples x n_features)
        if isinstance(shap_vals, list):
            # Older SHAP versions return [neg_class, pos_class]
            shap_array = shap_vals[1][0]
        else:
            shap_array = shap_vals[0]

        # 4. Build SHAP dict: feature_name → shap_value
        shap_dict = {
            feat: float(val)
            for feat, val in zip(self.feature_names, shap_array)
        }

        # 5. Get top N features by absolute SHAP value (most impactful)
        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:TOP_N_REASONS]

        # 6. Build human-readable reasons
        feature_values = X.iloc[0].to_dict()
        top_reasons = [
            self._build_reason_text(feat, val, feature_values.get(feat, -999))
            for feat, val in sorted_features
        ]

        # 7. Base score = expected value (average prediction across training)
        base_score = float(self.explainer.expected_value)
        if isinstance(base_score, (list, np.ndarray)):
            base_score = float(base_score[1])

        return {
            "fraud_score":     round(fraud_score, 4),
            "risk_level":      risk_level,
            "risk_action":     risk_action,
            "top_reasons":     top_reasons,
            "shap_values":     {k: round(v, 4) for k, v in sorted_features},
            "base_score":      round(base_score, 4),
            "feature_values":  {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in feature_values.items()
                                 if v != -999.0},
        }


# ──────────────────────────────────────────────────────────
# STANDALONE TEST
# Run: python src/explainer.py
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    explainer = FraudExplainer()

    # Simulate a suspicious transaction
    # (3am, large round amount, new device signals)
    suspicious_tx = {
        "TransactionID":  12345,
        "TransactionAmt": 500.00,       # large round number
        "TransactionDT":  10800,        # 3:00am (10800 seconds from midnight)
        "ProductCD":      0,
        "card1":          12345,
        "card4":          1,            # Visa
        "card6":          0,            # debit
        "P_emaildomain":  2,            # protonmail (encoded)
        "C1":             1,
        "C2":             1,
        "D1":             0,            # first transaction (new card)
        "D10":            0,            # new device
        "dist1":          500,          # 500km from billing address
    }

    result = explainer.explain_prediction(suspicious_tx)

    print("\n" + "=" * 60)
    print("  FRAUD EXPLANATION DEMO")
    print("=" * 60)
    print(f"\n  Fraud Score:  {result['fraud_score']} ({result['risk_level']})")
    print(f"  Action:       {result['risk_action']}")
    print(f"\n  Top Reasons:")
    for i, reason in enumerate(result["top_reasons"], 1):
        print(f"    {i}. {reason}")
    print(f"\n  Base Rate:    {result['base_score']} (avg fraud rate in training)")
    print("\n  Raw SHAP values:")
    print(json.dumps(result["shap_values"], indent=4))