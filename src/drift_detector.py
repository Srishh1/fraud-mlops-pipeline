"""
src/drift_detector.py
=====================
Drift detection engine using PSI + Evidently AI.

WHAT IS DATA DRIFT? (Plain English)
-------------------------------------
Your model was trained on January data. It's now June.
Has the incoming transaction data changed? Maybe:
  - A new payment method became popular (card type distribution shifted)
  - Fraud patterns changed (fraudsters now use smaller amounts)
  - A new merchant category emerged

If the data distribution changes significantly, your model's predictions
become unreliable — it's predicting based on patterns that no longer exist.
This is called DATA DRIFT.

TWO TOOLS IN THIS FILE:
  1. PSI (Population Stability Index) — our primary drift metric
     - Fast, simple, no external dependency
     - Industry standard at Razorpay, PayPal, banks globally
     - A single number per feature: < 0.1 stable, 0.1-0.2 warning, > 0.2 RETRAIN

  2. Evidently AI — our reporting tool
     - Generates a full HTML drift report with charts
     - Used for weekly drift reviews and audit logs
     - Heavier but more detailed than PSI alone

HOW THIS FITS INTO THE PIPELINE:
  Airflow DAG (Week 5) calls: check_drift() every day
    → if any feature PSI > 0.2: trigger retraining
    → else: skip, model is still good

WHY THIS MAKES YOUR PROJECT UNIQUE:
  Most student projects train a model and stop.
  Yours MONITORS itself and REACTS to change.
  That's what "self-healing" means. This file is the heart of it.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────

# PSI thresholds — industry standard values used by banks and fintechs
PSI_STABLE  = 0.1    # < 0.1:  no action needed
PSI_WARNING = 0.1    # 0.1-0.2: monitor closely
PSI_RETRAIN = 0.2    # > 0.2:  trigger retraining

# Features we monitor for drift
# WHY THESE SPECIFICALLY?
#   - TransactionAmt: most sensitive to fraud pattern changes
#   - Transaction_hour: time-of-day patterns shift seasonally
#   - card4, card6: payment method distribution changes over time
#   - C1, C2: velocity features — how often cards/emails are reused
#   - D1, D10: recency features — new vs returning customers
#   - dist1: geographic distribution changes
MONITORED_FEATURES = [
    "TransactionAmt",
    "Transaction_hour",
    "card1",
    "card4",
    "card6",
    "addr1",
    "dist1",
    "C1",
    "C2",
    "C6",
    "C13",
    "D1",
    "D10",
    "D15",
    "TransactionAmt_log",
]

PROJECT_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_STATS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reference_stats.json")
DRIFT_REPORT_DIR     = os.path.join(PROJECT_ROOT, "data", "processed", "drift_reports")
RAW_DATA_DIR         = os.path.join(PROJECT_ROOT, "data", "raw")

# ──────────────────────────────────────────────────────────
# CORE PSI CALCULATION
# ──────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray,
                buckets: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    FORMULA:
      PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    WHAT THIS MEASURES:
      How much has the distribution of a feature shifted between
      the training window (expected) and the current window (actual)?

    BUCKETS:
      We divide the expected distribution into 10 equal-frequency buckets
      (percentiles: 0-10%, 10-20%, ..., 90-100%).
      Then we count what % of actual data falls in each bucket.
      PSI measures the divergence between the two percentage vectors.

    Args:
        expected: array from training/reference period
        actual:   array from current production period
        buckets:  number of percentile buckets (10 is standard)

    Returns:
        PSI score (float, always >= 0)

    PSI INTERPRETATION:
        < 0.1:  No significant shift. Model stable.
        0.1-0.2: Some shift. Monitor closely.
        > 0.2:  Significant shift. Retrain recommended.
        > 0.5:  Extreme shift. Something is very wrong with data pipeline.
    """
    # Remove NaN and the missing sentinel value (-999)
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    expected = expected[expected != -999]
    actual   = actual[actual != -999]

    if len(expected) == 0 or len(actual) == 0:
        logger.warning("Empty array passed to compute_psi — returning 0")
        return 0.0

    # Build bucket breakpoints from the EXPECTED distribution
    # WHY EXPECTED? The expected distribution defines "what normal looks like"
    # We always measure actual against that baseline
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates from skewed distributions

    if len(breakpoints) < 2:
        logger.warning("Could not create enough breakpoints — distribution too skewed")
        return 0.0

    # Count how many observations fall in each bucket
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Convert to percentages, adding epsilon to avoid division by zero / log(0)
    # WHY EPSILON? log(0) = -infinity which breaks the formula
    epsilon = 1e-6
    n_exp = len(expected) + epsilon * len(expected_counts)
    n_act = len(actual)   + epsilon * len(actual_counts)

    expected_pct = (expected_counts + epsilon) / n_exp
    actual_pct   = (actual_counts   + epsilon) / n_act

    # PSI formula
    psi_per_bucket = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = float(np.sum(psi_per_bucket))

    return max(0.0, psi)  # PSI is always non-negative


def compute_psi_all_features(reference_df: pd.DataFrame,
                              current_df: pd.DataFrame,
                              features: list = None) -> dict:
    """
    Compute PSI for every monitored feature.

    Args:
        reference_df: DataFrame from training / last stable period
        current_df:   DataFrame from current production window
        features:     list of feature names to check (defaults to MONITORED_FEATURES)

    Returns:
        dict of {feature_name: psi_score}
    """
    if features is None:
        features = MONITORED_FEATURES

    psi_scores = {}
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            logger.debug(f"Feature '{feature}' not found in one of the DataFrames — skipping")
            continue

        psi = compute_psi(
            reference_df[feature].values,
            current_df[feature].values,
        )
        psi_scores[feature] = round(psi, 4)
        logger.debug(f"  PSI {feature}: {psi:.4f}")

    return psi_scores


# ──────────────────────────────────────────────────────────
# REFERENCE STATISTICS
# ──────────────────────────────────────────────────────────

def build_reference_stats(df: pd.DataFrame,
                           features: list = None) -> dict:
    """
    Compute and save reference statistics from training data.

    WHAT ARE REFERENCE STATS?
    They're a snapshot of what your training data looked like.
    We store: mean, std, percentiles, and the raw sample for each feature.
    Later, when new data arrives, we compare it against this reference.

    WHY SAVE TO DISK?
    The Airflow DAG runs in a separate process. It needs access to the
    reference stats without re-loading the full training dataset.

    WHEN TO RUN THIS:
    - After initial training (Week 1 done → run this once)
    - After every retrain (to update the baseline)
    - Called automatically by train.py in the full pipeline (Week 5)
    """
    if features is None:
        features = MONITORED_FEATURES

    stats = {
        "created_at": datetime.utcnow().isoformat(),
        "n_samples": len(df),
        "features": {}
    }

    for feature in features:
        if feature not in df.columns:
            continue
        col = df[feature].replace(-999, np.nan).dropna()
        if len(col) == 0:
            continue

        # Store key statistics
        stats["features"][feature] = {
            "mean":   round(float(col.mean()), 4),
            "std":    round(float(col.std()), 4),
            "min":    round(float(col.min()), 4),
            "max":    round(float(col.max()), 4),
            "p25":    round(float(col.quantile(0.25)), 4),
            "p50":    round(float(col.quantile(0.50)), 4),
            "p75":    round(float(col.quantile(0.75)), 4),
            "p95":    round(float(col.quantile(0.95)), 4),
            "null_rate": round(float((df[feature] == -999).mean()), 4),
            # Store a sample of 5000 values for PSI computation
            # 5000 is enough for stable percentile estimates
            "sample": col.sample(min(5000, len(col)),
                                  random_state=42).tolist(),
        }

    # Save to disk
    os.makedirs(os.path.dirname(REFERENCE_STATS_PATH), exist_ok=True)
    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Reference stats saved → {REFERENCE_STATS_PATH}")
    print(f"   Features tracked: {len(stats['features'])}")
    return stats


def load_reference_stats() -> dict:
    """Load previously saved reference statistics."""
    if not os.path.exists(REFERENCE_STATS_PATH):
        raise FileNotFoundError(
            f"Reference stats not found at {REFERENCE_STATS_PATH}.\n"
            "Run: build_reference_stats(training_df) first."
        )
    with open(REFERENCE_STATS_PATH) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────
# SIMULATED DRIFT (for demo/testing without live production data)
# ──────────────────────────────────────────────────────────

def simulate_drift(reference_df: pd.DataFrame,
                   drift_intensity: str = "moderate") -> pd.DataFrame:
    """
    Simulate data drift for testing and demo purposes.

    WHY DO WE NEED THIS?
    In a real deployment, drift happens naturally over months.
    We can't wait months to demo the drift detection system.
    So we artificially shift the data distribution to test our detector.

    This is standard practice when building and testing drift detection systems.
    You simulate drift to verify your system catches it before real drift happens.

    DRIFT SCENARIOS:
      "none"     → identical to reference (PSI ≈ 0)
      "slight"   → small shift (PSI ≈ 0.05-0.1, no alarm)
      "moderate" → medium shift (PSI ≈ 0.1-0.2, warning)
      "severe"   → large shift (PSI > 0.2, triggers retraining)
    """
    drifted = reference_df.copy()
    rng = np.random.RandomState(42)

    shift_config = {
        "none":     {"amt_mult": 1.0,  "hour_shift": 0,  "noise": 0.0},
        "slight":   {"amt_mult": 1.15, "hour_shift": 1,  "noise": 0.05},
        "moderate": {"amt_mult": 1.5,  "hour_shift": 3,  "noise": 0.15},
        "severe":   {"amt_mult": 3.0,  "hour_shift": 8,  "noise": 0.40},
    }

    config = shift_config.get(drift_intensity, shift_config["moderate"])

    # Shift transaction amounts (fraud amounts tend to increase over time)
    if "TransactionAmt" in drifted.columns:
        valid_mask = drifted["TransactionAmt"] != -999
        drifted.loc[valid_mask, "TransactionAmt"] *= config["amt_mult"]
        drifted.loc[valid_mask, "TransactionAmt"] += \
            rng.normal(0, config["noise"] * drifted.loc[valid_mask, "TransactionAmt"].std(),
                       valid_mask.sum())
        if "TransactionAmt_log" in drifted.columns:
            drifted.loc[valid_mask, "TransactionAmt_log"] = \
                np.log1p(drifted.loc[valid_mask, "TransactionAmt"])

    # Shift transaction hours (fraud timing patterns change)
    if "Transaction_hour" in drifted.columns:
        valid_mask = drifted["Transaction_hour"] != -999
        drifted.loc[valid_mask, "Transaction_hour"] = \
            (drifted.loc[valid_mask, "Transaction_hour"] + config["hour_shift"]) % 24

    # Add noise to card distribution
    if "card1" in drifted.columns:
        valid_mask = drifted["card1"] != -999
        noise = rng.normal(0, config["noise"] * 1000, valid_mask.sum())
        drifted.loc[valid_mask, "card1"] += noise

    # Shift distance feature (geographic distribution change)
    if "dist1" in drifted.columns:
        valid_mask = drifted["dist1"] != -999
        drifted.loc[valid_mask, "dist1"] *= (1 + config["noise"] * 2)

    print(f"✅ Simulated '{drift_intensity}' drift applied to {len(drifted):,} samples")
    return drifted


# ──────────────────────────────────────────────────────────
# EVIDENTLY REPORT
# ──────────────────────────────────────────────────────────

def generate_evidently_report(reference_df: pd.DataFrame,
                               current_df: pd.DataFrame,
                               features: list = None,
                               report_name: str = None) -> str:
    """
    Generate a full HTML drift report using Evidently AI.

    WHAT IS EVIDENTLY?
    Evidently is an open-source ML monitoring library.
    It generates beautiful HTML reports showing:
      - Feature drift for every column (with charts)
      - Dataset-level drift summary
      - Data quality metrics
      - Distribution comparison plots

    WHY USE EVIDENTLY IN ADDITION TO PSI?
    PSI gives you ONE NUMBER per feature — great for automation.
    Evidently gives you FULL VISUAL REPORTS — great for humans reviewing drift.
    In production: Airflow uses PSI to decide, your team reads Evidently reports.

    The generated HTML file is saved to data/processed/drift_reports/
    and can be served as a static file or stored in S3.

    Args:
        reference_df: training / baseline data
        current_df:   current production window data
        features:     which columns to analyze
        report_name:  filename suffix (defaults to timestamp)

    Returns:
        path to generated HTML report
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import (
            DatasetDriftMetric,
            DatasetMissingValuesMetric,
            ColumnDriftMetric,
        )
    except ImportError:
        raise ImportError(
            "Evidently not installed. Run: pip install evidently==0.4.30"
        )

    if features is None:
        features = MONITORED_FEATURES

    # Filter to features that exist in both DataFrames
    common_features = [f for f in features
                       if f in reference_df.columns and f in current_df.columns]

    ref = reference_df[common_features].copy()
    cur = current_df[common_features].copy()

    # Replace our missing sentinel with actual NaN so Evidently handles it correctly
    ref = ref.replace(-999, np.nan)
    cur = cur.replace(-999, np.nan)

    # Build the report
    # DataDriftPreset = full suite of drift tests for every column
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=ref, current_data=cur)

    # Save to disk
    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)
    timestamp = report_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(DRIFT_REPORT_DIR, f"drift_report_{timestamp}.html")
    report.save_html(report_path)

    print(f"✅ Evidently drift report saved → {report_path}")
    return report_path


# ──────────────────────────────────────────────────────────
# MAIN DRIFT CHECK FUNCTION
# Called by Airflow DAG in Week 5
# ──────────────────────────────────────────────────────────

def check_drift(current_df: pd.DataFrame,
                features: list = None,
                generate_report: bool = True) -> dict:
    """
    Main entry point for drift checking.
    Called daily by the Airflow DAG.

    WHAT THIS DOES:
      1. Loads reference stats from disk (saved during training)
      2. Computes PSI for every monitored feature
      3. Determines overall drift status
      4. Optionally generates an Evidently HTML report
      5. Returns a structured result dict

    RETURN VALUE:
    {
        "drift_detected": True/False,   ← Airflow branches on this
        "max_psi": 0.34,
        "drifted_features": ["TransactionAmt", "dist1"],
        "psi_scores": {"TransactionAmt": 0.34, "dist1": 0.28, ...},
        "status": "drifted",            ← stable/warning/drifted
        "recommendation": "Retrain...",
        "report_path": "data/processed/drift_reports/drift_report_20240615.html",
        "checked_at": "2024-06-15T10:30:00"
    }

    The Airflow DAG reads "drift_detected" and branches:
        True  → trigger retraining
        False → skip, log "model stable"
    """
    print("\n🔍 Running drift check...")
    print(f"   Current data: {len(current_df):,} samples")

    # Load reference data
    ref_stats = load_reference_stats()
    print(f"   Reference data: {ref_stats['n_samples']:,} samples (from {ref_stats['created_at'][:10]})")

    # Rebuild reference DataFrame from stored samples
    ref_data = {}
    for feature, stats in ref_stats["features"].items():
        ref_data[feature] = stats["sample"]

    # Pad shorter arrays to the same length for DataFrame construction
    max_len = max(len(v) for v in ref_data.values())
    for feature in ref_data:
        arr = ref_data[feature]
        if len(arr) < max_len:
            ref_data[feature] = arr + [np.nan] * (max_len - len(arr))

    reference_df = pd.DataFrame(ref_data)

    # Compute PSI for all features
    if features is None:
        features = [f for f in MONITORED_FEATURES if f in current_df.columns]

    psi_scores = compute_psi_all_features(reference_df, current_df, features)

    if not psi_scores:
        print("  ⚠️  No features could be compared — check data format")
        return {"drift_detected": False, "error": "no_features_compared"}

    # Determine status
    max_psi = max(psi_scores.values())
    drifted_features = [f for f, psi in psi_scores.items() if psi > PSI_RETRAIN]
    warning_features = [f for f, psi in psi_scores.items() if PSI_WARNING <= psi <= PSI_RETRAIN]

    drift_detected = len(drifted_features) > 0

    if drift_detected:
        status = "drifted"
        recommendation = (
            f"RETRAIN REQUIRED. {len(drifted_features)} feature(s) exceed PSI threshold "
            f"({PSI_RETRAIN}): {', '.join(drifted_features)}"
        )
    elif warning_features:
        status = "warning"
        recommendation = (
            f"Monitor closely. {len(warning_features)} feature(s) show moderate drift: "
            f"{', '.join(warning_features)}"
        )
    else:
        status = "stable"
        recommendation = "All features stable. No action required."

    print(f"\n   📊 PSI Results:")
    for feature, psi in sorted(psi_scores.items(), key=lambda x: -x[1]):
        flag = "🔴 RETRAIN" if psi > PSI_RETRAIN else "🟡 WARNING" if psi > PSI_WARNING else "🟢"
        print(f"      {flag} {feature}: {psi:.4f}")

    print(f"\n   Status: {status.upper()}")
    print(f"   Max PSI: {max_psi:.4f}")
    print(f"   Drift detected: {drift_detected}")

    # Generate Evidently report
    report_path = None
    if generate_report:
        try:
            report_path = generate_evidently_report(
                reference_df, current_df, features
            )
        except Exception as e:
            print(f"  ⚠️  Evidently report failed (non-critical): {e}")

    result = {
        "drift_detected":   drift_detected,
        "status":           status,
        "max_psi":          round(max_psi, 4),
        "psi_scores":       psi_scores,
        "drifted_features": drifted_features,
        "warning_features": warning_features,
        "recommendation":   recommendation,
        "report_path":      report_path,
        "checked_at":       datetime.utcnow().isoformat(),
    }

    # Save result to disk so Airflow can read it
    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)
    result_path = os.path.join(DRIFT_REPORT_DIR, "latest_drift_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n   ✅ Result saved → {result_path}")
    return result


# ──────────────────────────────────────────────────────────
# FEATURE ENGINEERING (shared with train.py)
# Needed to prepare raw data for drift comparison
# ──────────────────────────────────────────────────────────

def engineer_features_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as train.py so drift comparison
    is apples-to-apples.

    WHY REPEAT THIS HERE?
    The drift detector operates on engineered features — the same ones
    the model sees. If we compare raw TransactionAmt but the model
    sees log(TransactionAmt), we'd miss drift in the actual model input.
    """
    df = df.copy()

    if "TransactionDT" in df.columns:
        df["Transaction_hour"] = (df["TransactionDT"] // 3600) % 24
        df["Transaction_day"]  = (df["TransactionDT"] // (3600 * 24)) % 7

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].clip(lower=0))
        df["TransactionAmt_rounded"] = (df["TransactionAmt"] % 1 == 0).astype(int)

    # Encode ALL categorical/string columns
    # M1-M9 contain 'T'/'F' strings — must be encoded before parquet save
    categorical_cols = ["ProductCD", "card4", "card6", "P_emaildomain",
                        "R_emaildomain", "M1", "M2", "M3", "M4", "M5",
                        "M6", "M7", "M8", "M9"]
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            df[col] = le.fit_transform(df[col])

    # Catch any remaining object columns pyarrow cannot serialize
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown").astype(str)
        try:
            df[col] = le.fit_transform(df[col])
        except Exception:
            df[col] = -999

    df = df.fillna(-999)
    return df


# ──────────────────────────────────────────────────────────
# STANDALONE DEMO — python src/drift_detector.py
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  DRIFT DETECTION ENGINE — DEMO")
    print("=" * 60)

    # Load training data to build reference
    transaction_path = os.path.join(RAW_DATA_DIR, "train_transaction.csv")
    identity_path    = os.path.join(RAW_DATA_DIR, "train_identity.csv")

    if not os.path.exists(transaction_path):
        print(f"\n❌ Training data not found at {transaction_path}")
        print("   Run src/train.py first, then this script.")
        sys.exit(1)

    print("\n📂 Loading training data for reference...")
    df = pd.read_csv(transaction_path, low_memory=False)
    if os.path.exists(identity_path):
        df_id = pd.read_csv(identity_path, low_memory=False)
        df = df.merge(df_id, on="TransactionID", how="left")

    df = engineer_features_for_drift(df)

    # Use first 80% as reference (matches train/val split)
    split = int(len(df) * 0.8)
    reference_df = df.iloc[:split]
    holdout_df   = df.iloc[split:]

    print(f"   Reference: {len(reference_df):,} samples")
    print(f"   Holdout:   {len(holdout_df):,} samples")

    # Build and save reference stats
    print("\n📊 Building reference statistics...")
    build_reference_stats(reference_df)

    # ── Scenario 1: Stable data (holdout = similar to training) ──
    print("\n" + "─" * 50)
    print("SCENARIO 1: Stable data (real holdout)")
    print("─" * 50)
    result_stable = check_drift(holdout_df, generate_report=False)

    # ── Scenario 2: Simulated severe drift ──────────────────
    print("\n" + "─" * 50)
    print("SCENARIO 2: Simulated severe drift")
    print("─" * 50)
    drifted_df = simulate_drift(holdout_df, drift_intensity="severe")
    result_drifted = check_drift(drifted_df, generate_report=True)

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DRIFT DETECTION DEMO SUMMARY")
    print("=" * 60)
    print(f"\n  Stable scenario:  status={result_stable['status']}, "
          f"max_psi={result_stable['max_psi']:.4f}, "
          f"retrain={result_stable['drift_detected']}")
    print(f"  Drifted scenario: status={result_drifted['status']}, "
          f"max_psi={result_drifted['max_psi']:.4f}, "
          f"retrain={result_drifted['drift_detected']}")

    if result_drifted["drift_detected"] and not result_stable["drift_detected"]:
        print("\n  ✅ Drift detector works correctly!")
        print("     - Stable data: no alarm (good)")
        print("     - Drifted data: alarm triggered (good)")
    else:
        print("\n  ⚠️  Check drift thresholds — results unexpected")

    if result_drifted.get("report_path"):
        print(f"\n  📄 Evidently report: {result_drifted['report_path']}")
        print("     Open in browser to see full distribution analysis")