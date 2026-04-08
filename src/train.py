"""
src/train.py
============
Baseline XGBoost fraud detection model for the IEEE-CIS dataset.

WHAT THIS FILE DOES (plain English):
  1. Loads and cleans the raw Kaggle transaction data
  2. Engineers features (creates new useful columns from existing ones)
  3. Trains an XGBoost classifier to predict fraud (isFraud = 1)
  4. Evaluates it — we want AUC-ROC > 0.88
  5. Saves the trained model + feature list to disk (ready for Week 3 MLflow)

WHY XGBoost for fraud detection?
  - Handles imbalanced data well (fraud is ~3.5% of transactions)
  - Fast to train on CPU — no GPU needed
  - Built-in feature importance — pairs perfectly with SHAP later
  - Industry standard: Razorpay, PayPal, Stripe all use gradient boosting variants
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import precision_recall_curve
import json


# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────

# Paths — relative to project root
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"

TRANSACTION_FILE = os.path.join(RAW_DATA_DIR, "train_transaction.csv")
IDENTITY_FILE = os.path.join(RAW_DATA_DIR, "train_identity.csv")

MODEL_OUTPUT = os.path.join(MODEL_DIR, "fraud_model_v1.joblib")
FEATURE_LIST_OUTPUT = os.path.join(MODEL_DIR, "feature_names.json")

# Random seed — always set this so your results are reproducible
RANDOM_STATE = 42

# XGBoost hyperparameters
# WHY THESE VALUES?
#   scale_pos_weight: dataset is ~96.5% non-fraud, ~3.5% fraud
#   so we tell XGBoost: "treat each fraud sample as if it's 28x more important"
#   This prevents the model from cheating by predicting "not fraud" every time
XGBOOST_PARAMS = {
    "n_estimators": 500,          # number of trees — more = better but slower
    "max_depth": 6,               # how deep each tree goes (6-8 is sweet spot)
    "learning_rate": 0.05,        # step size — smaller = slower but more accurate
    "subsample": 0.8,             # use 80% of rows per tree (prevents overfitting)
    "colsample_bytree": 0.8,      # use 80% of features per tree
    "scale_pos_weight": 28,       # handles class imbalance (96.5/3.5 ≈ 28)
    "use_label_encoder": False,
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,                 # use all CPU cores
    "tree_method": "hist",        # fast CPU training method
}


# ──────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ──────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load and merge the two Kaggle IEEE-CIS CSV files.

    WHY TWO FILES?
    Kaggle split the data into:
    - train_transaction.csv: the actual payment info (card, amount, device)
    - train_identity.csv: who made the transaction (browser, OS, device type)
    We join them on TransactionID.
    """
    print("📂 Loading raw data...")

    if not os.path.exists(TRANSACTION_FILE):
        raise FileNotFoundError(
            f"\n❌ File not found: {TRANSACTION_FILE}\n"
            "Please download the IEEE-CIS dataset from Kaggle:\n"
            "  kaggle competitions download -c ieee-fraud-detection\n"
            "Then unzip into data/raw/\n"
        )

    # Load with low_memory=False to avoid dtype warning on mixed columns
    df_trans = pd.read_csv(TRANSACTION_FILE, low_memory=False)
    print(f"  ✅ Transactions loaded: {df_trans.shape}")

    if os.path.exists(IDENTITY_FILE):
        df_id = pd.read_csv(IDENTITY_FILE, low_memory=False)
        df = df_trans.merge(df_id, on="TransactionID", how="left")
        print(f"  ✅ Identity merged: {df.shape}")
    else:
        print("  ⚠️  Identity file not found, using transaction data only")
        df = df_trans

    return df


# ──────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features and clean the data.

    WHAT IS FEATURE ENGINEERING?
    Raw data often isn't enough. We CREATE new columns that help the model
    detect patterns. For example: the hour of a transaction is more useful
    than the raw timestamp, because fraud spikes at 3am.

    WHAT IS LABEL ENCODING?
    XGBoost only understands numbers, not strings.
    So we convert "Chrome" → 0, "Firefox" → 1, "Safari" → 2 etc.
    """
    print("\n🔧 Engineering features...")

    df = df.copy()

    # ── Time features ──
    # TransactionDT is seconds since some reference point (not unix timestamp)
    # We extract cyclic patterns from it
    df["Transaction_hour"] = (df["TransactionDT"] // 3600) % 24
    df["Transaction_day"] = (df["TransactionDT"] // (3600 * 24)) % 7

    # ── Amount features ──
    # Log transform: reduces the effect of extreme outliers
    # e.g., a $50,000 transaction doesn't dominate over $50 ones
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Amount rounded (fraudsters often use round numbers like $500.00)
    df["TransactionAmt_rounded"] = (df["TransactionAmt"] % 1 == 0).astype(int)

    # ── Encode categorical columns ──
    # These are string columns XGBoost can't read directly
    categorical_cols = ["ProductCD", "card4", "card6", "P_emaildomain",
                        "R_emaildomain", "M1", "M2", "M3", "M4", "M5",
                        "M6", "M7", "M8", "M9"]

    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            # Fill nulls with "unknown" before encoding
            df[col] = df[col].fillna("unknown").astype(str)
            df[col] = le.fit_transform(df[col])

    print(f"  ✅ Features engineered. Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────
# STEP 3: PREPARE FEATURES + TARGET
# ──────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """
    Split the dataframe into X (features) and y (target).

    WHY DROP THESE COLUMNS?
    - isFraud: this IS our target — can't use it to predict itself
    - TransactionID: just an ID, no signal
    - TransactionDT: we extracted useful parts of it already
    """
    # Target
    y = df["isFraud"].astype(int)

    # Drop columns that shouldn't be features
    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill remaining NaN values with -999
    # WHY -999? XGBoost handles missing values natively, but we want
    # consistency. -999 is a sentinel that says "this was missing"
    X = X.fillna(-999)

    # Keep only numeric columns (safety net)
    X = X.select_dtypes(include=[np.number])

    print(f"\n📊 Feature matrix: {X.shape}")
    print(f"   Target distribution:\n{y.value_counts(normalize=True).round(3)}")

    return X, y


# ──────────────────────────────────────────────────────────
# STEP 4: TRAIN + EVALUATE
# ──────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost with early stopping.

    WHAT IS EARLY STOPPING?
    We tell XGBoost: "keep training more trees, but STOP if the validation
    AUC hasn't improved in the last 50 rounds." This prevents overfitting.
    """
    print("\n🚀 Training XGBoost model...")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")

    model = xgb.XGBClassifier(
        **XGBOOST_PARAMS,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,  # print progress every 100 trees
    )

    print(f"\n   Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, X_val, y_val) -> dict:
    """
    Evaluate the model and return a metrics dictionary.

    METRICS EXPLAINED:
    - AUC-ROC: "how good is the model at RANKING fraud vs not-fraud?"
               0.5 = random guessing, 1.0 = perfect. We want > 0.88
    - AUC-PR:  "how precise are we when we do predict fraud?"
               More informative than AUC-ROC on imbalanced datasets
    - The classification report shows precision, recall, F1 at threshold 0.5
    """
    print("\n📈 Evaluating model...")

    y_pred_proba = model.predict_proba(X_val)[:, 1]  # fraud probability
    y_pred = (y_pred_proba > 0.5).astype(int)        # binary prediction

    auc_roc = roc_auc_score(y_val, y_pred_proba)
    auc_pr = average_precision_score(y_val, y_pred_proba)

    print(f"\n   🎯 AUC-ROC:       {auc_roc:.4f}  (target: > 0.88)")
    print(f"   🎯 AUC-PR:        {auc_pr:.4f}")
    print(f"\n   Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    print(f"\n   Classification Report:\n{classification_report(y_val, y_pred)}")

    if auc_roc >= 0.88:
        print("\n   ✅ AUC-ROC target achieved!")
    else:
        print("\n   ⚠️  AUC-ROC below target. Consider tuning hyperparameters.")

    return {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "best_iteration": model.best_iteration,
    }

def find_optimal_threshold(model, X_val, y_val):
    """
    Find the threshold that maximizes F1 score.
    In production, this gets tuned to business requirements instead.
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    
    print(f"\n   📍 Optimal threshold (max F1): {best_threshold:.3f}")
    print(f"      Precision at optimal: {precisions[best_idx]:.3f}")
    print(f"      Recall at optimal:    {recalls[best_idx]:.3f}")
    print(f"      F1 at optimal:        {f1_scores[best_idx]:.3f}")
    
    return best_threshold

# ──────────────────────────────────────────────────────────
# STEP 5: SAVE MODEL
# ──────────────────────────────────────────────────────────

def save_model(model, feature_names: list):
    """
    Save the trained model and feature list to disk.

    WHY SAVE FEATURE NAMES SEPARATELY?
    When we serve predictions in production (Week 2), incoming data must have
    EXACTLY the same columns in the same order as training data.
    Storing feature names lets us validate this before predicting.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_OUTPUT)
    print(f"\n💾 Model saved → {MODEL_OUTPUT}")

    with open(FEATURE_LIST_OUTPUT, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"💾 Feature names saved → {FEATURE_LIST_OUTPUT}")


# ──────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def run_training_pipeline():
    """
    Orchestrates the full training pipeline end-to-end.
    This is what gets called by Airflow in Week 5 as well.
    """
    print("=" * 60)
    print("  FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Feature Engineering
    df = engineer_features(df)

    # 3. Prepare X, y
    X, y = prepare_features(df)

    # 4. Train/Val split
    # WHY NOT random shuffle? Fraud data is time-ordered.
    # In production, we'd split by time. For baseline, stratified split is fine.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,         # ensures both splits have ~3.5% fraud
        random_state=RANDOM_STATE,
    )

    # 5. Train
    model = train_model(X_train, y_train, X_val, y_val)

    # 6. Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # 7. Save
    save_model(model, list(X.columns))

    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE")
    print(f"  AUC-ROC: {metrics['auc_roc']}")
    print("=" * 60)

    return model, metrics


if __name__ == "__main__":
    run_training_pipeline()
