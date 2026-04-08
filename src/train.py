"""
src/train.py
============
XGBoost fraud classifier — Week 3: MLflow integrated version.

WHAT CHANGED FROM WEEK 2:
  - Every training run is logged to MLflow (params, metrics, artifacts)
  - Model is registered in MLflow Model Registry after training
  - Model is promoted to "Production" stage automatically if AUC > threshold
  - The hardcoded .joblib file still gets saved as fallback for local dev

WHAT IS MLFLOW? (Plain English)
---------------------------------
Imagine you're training a model and you tweak it 20 times over 2 weeks.
Without MLflow: "Which run had AUC 0.946? What params did I use? Where's that model?"
With MLflow: every run is logged. You can compare them, reproduce any one,
and promote the best one to "Production" with a single click or API call.

MLflow has 4 components:
  1. Tracking     → logs params, metrics, plots for each run
  2. Projects     → reproducible run packaging (we skip this)
  3. Models       → standard model format (works with sklearn, xgboost, etc.)
  4. Registry     → version control for models (Staging → Production → Archived)

HOW THE REGISTRY WORKS:
  train.py runs → logs model as "fraud-detector" version N
               → if AUC > 0.88, promotes to Production stage
  api.py starts → loads model from registry by stage: "Production"
               → no hardcoded path, always gets latest promoted model

WHY THIS MATTERS FOR INTERVIEWS:
  "How do you manage model versions in production?"
  This is asked at every MLOps interview. MLflow Registry is the industry answer.
"""

import os
import json
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from mlflow.tracking import MlflowClient


# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────

RAW_DATA_DIR      = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR         = "models"

TRANSACTION_FILE  = os.path.join(RAW_DATA_DIR, "train_transaction.csv")
IDENTITY_FILE     = os.path.join(RAW_DATA_DIR, "train_identity.csv")

MODEL_OUTPUT      = os.path.join(MODEL_DIR, "fraud_model_v1.joblib")
FEATURE_LIST_OUTPUT = os.path.join(MODEL_DIR, "feature_names.json")

RANDOM_STATE      = 42
AUC_PROMOTION_THRESHOLD = 0.88   # model must beat this to be promoted to Production

# MLflow config
# MLFLOW_TRACKING_URI: where to store runs.
#   "mlruns"  → local folder (default, no server needed)
#   "http://localhost:5000" → MLflow server (docker-compose)
#   Set via env var so docker-compose can override it
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT   = "fraud-detection-baseline"
REGISTERED_MODEL_NAME = "fraud-detector"

XGBOOST_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  28,
    "use_label_encoder": False,
    "eval_metric":       "auc",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "tree_method":       "hist",
}


# ──────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# (unchanged from Week 1 — keeping it here for self-contained file)
# ──────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("📂 Loading raw data...")
    if not os.path.exists(TRANSACTION_FILE):
        raise FileNotFoundError(
            f"\n❌ File not found: {TRANSACTION_FILE}\n"
            "Download from Kaggle: ieee-fraud-detection\n"
        )
    df_trans = pd.read_csv(TRANSACTION_FILE, low_memory=False)
    print(f"  ✅ Transactions loaded: {df_trans.shape}")

    if os.path.exists(IDENTITY_FILE):
        df_id = pd.read_csv(IDENTITY_FILE, low_memory=False)
        df = df_trans.merge(df_id, on="TransactionID", how="left")
        print(f"  ✅ Identity merged: {df.shape}")
    else:
        df = df_trans
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 Engineering features...")
    df = df.copy()
    df["Transaction_hour"]     = (df["TransactionDT"] // 3600) % 24
    df["Transaction_day"]      = (df["TransactionDT"] // (3600 * 24)) % 7
    df["TransactionAmt_log"]   = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_rounded"] = (df["TransactionAmt"] % 1 == 0).astype(int)

    categorical_cols = ["ProductCD", "card4", "card6", "P_emaildomain",
                        "R_emaildomain", "M1", "M2", "M3", "M4", "M5",
                        "M6", "M7", "M8", "M9"]
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            df[col] = le.fit_transform(df[col])

    print(f"  ✅ Features engineered. Shape: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame):
    y = df["isFraud"].astype(int)
    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.fillna(-999)
    X = X.select_dtypes(include=[np.number])
    print(f"\n📊 Feature matrix: {X.shape}")
    print(f"   Target distribution:\n{y.value_counts(normalize=True).round(3)}")
    return X, y


# ──────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val):
    print("\n🚀 Training XGBoost model...")
    model = xgb.XGBClassifier(**XGBOOST_PARAMS, early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print(f"   Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, X_val, y_val) -> dict:
    print("\n📈 Evaluating model...")
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba > 0.5).astype(int)

    auc_roc = roc_auc_score(y_val, y_proba)
    auc_pr  = average_precision_score(y_val, y_proba)

    print(f"\n   🎯 AUC-ROC: {auc_roc:.4f}  (target: > {AUC_PROMOTION_THRESHOLD})")
    print(f"   🎯 AUC-PR:  {auc_pr:.4f}")
    print(f"\n   Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    print(f"\n   Classification Report:\n{classification_report(y_val, y_pred)}")

    return {
        "auc_roc":        round(auc_roc, 4),
        "auc_pr":         round(auc_pr, 4),
        "best_iteration": int(model.best_iteration),
        "n_estimators":   int(model.best_iteration + 1),
    }


# ──────────────────────────────────────────────────────────
# MLFLOW LOGGING
# ──────────────────────────────────────────────────────────

def log_to_mlflow(model, metrics: dict, feature_names: list,
                   X_val, y_val) -> str:
    """
    Log everything to MLflow and register the model.

    WHAT GETS LOGGED:
      Params  → hyperparameters (learning_rate, max_depth, etc.)
      Metrics → AUC-ROC, AUC-PR, best_iteration
      Artifacts → feature_names.json, feature_importance plot
      Model   → the XGBoost model itself (in MLflow format)

    WHAT IS THE MODEL REGISTRY?
      After logging, we "register" the model — this creates a versioned
      entry under the name "fraud-detector". Versions are numbered
      automatically: v1, v2, v3...
      We then set the stage:
        Staging    → tested, not yet in production
        Production → live, what the API loads
        Archived   → old versions kept for rollback

    RETURNS: the MLflow run_id (used for tracking/rollback)
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n📊 MLflow run started: {run_id}")

        # ── 1. Log hyperparameters ────────────────────────
        # WHY LOG PARAMS? So you can reproduce any run exactly.
        # MLflow stores these and lets you compare across runs.
        mlflow.log_params({
            "n_estimators":     XGBOOST_PARAMS["n_estimators"],
            "max_depth":        XGBOOST_PARAMS["max_depth"],
            "learning_rate":    XGBOOST_PARAMS["learning_rate"],
            "subsample":        XGBOOST_PARAMS["subsample"],
            "colsample_bytree": XGBOOST_PARAMS["colsample_bytree"],
            "scale_pos_weight": XGBOOST_PARAMS["scale_pos_weight"],
            "random_state":     RANDOM_STATE,
            "n_features":       len(feature_names),
            "dataset":          "ieee-cis-fraud-detection",
        })
        print("  ✅ Params logged")

        # ── 2. Log metrics ────────────────────────────────
        mlflow.log_metrics({
            "auc_roc":        metrics["auc_roc"],
            "auc_pr":         metrics["auc_pr"],
            "best_iteration": metrics["best_iteration"],
        })
        print("  ✅ Metrics logged")

        # ── 3. Log feature names as artifact ──────────────
        # Artifacts = any file you want to store alongside the run
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(FEATURE_LIST_OUTPUT, "w") as f:
            json.dump(feature_names, f, indent=2)
        mlflow.log_artifact(FEATURE_LIST_OUTPUT, artifact_path="model_assets")
        print("  ✅ Feature names artifact logged")

        # ── 4. Log feature importance plot ────────────────
        try:
            import matplotlib.pyplot as plt
            importance = model.feature_importances_
            top_idx = importance.argsort()[-20:][::-1]
            top_names  = [feature_names[i] for i in top_idx]
            top_scores = importance[top_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_names)), top_scores, color="#3498db")
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names, fontsize=9)
            ax.set_xlabel("Feature Importance (gain)")
            ax.set_title("Top 20 Features — XGBoost Fraud Classifier")
            ax.invert_yaxis()
            plt.tight_layout()

            plot_path = os.path.join(MODEL_DIR, "feature_importance.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="plots")
            print("  ✅ Feature importance plot logged")
        except Exception as e:
            print(f"  ⚠️  Could not save feature importance plot: {e}")

        # ── 5. Log the model to MLflow Model Registry ─────
        # mlflow.xgboost.log_model() saves the model in MLflow's
        # standard format AND registers it in the model registry.
        #
        # registered_model_name = the name that appears in the registry UI
        # artifact_path = folder inside the run where model is stored
        #
        # After this, the model is addressable as:
        #   models:/fraud-detector/1    (version 1)
        #   models:/fraud-detector/Production  (current production)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="xgboost_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_val.iloc[:1],
        )
        print(f"  ✅ Model registered as '{REGISTERED_MODEL_NAME}'")

    return run_id


def promote_model_to_production(run_id: str, auc_roc: float):
    """
    Promote the latest model version to Production stage if AUC passes threshold.

    WHAT IS MODEL PROMOTION?
    Think of it like a job interview:
      - Model trains → gets registered as version N (the "candidate")
      - If AUC > threshold → promoted to Production (the "hire")
      - Old Production version → archived (the "retired employee")
      - The API always loads "Production" → automatic rollout of new model

    WHAT HAPPENS WITHOUT THIS?
    If you hardcode the model path ("models/fraud_model_v1.joblib"),
    deploying a new model requires a code change + redeploy.
    With the registry, retraining + promotion = automatic rollout. Zero downtime.
    This is what "self-healing" means at the pipeline level.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get the latest version of our registered model
    versions = client.get_latest_versions(
        REGISTERED_MODEL_NAME, stages=["None", "Staging"]
    )

    if not versions:
        print("  ⚠️  No model versions found to promote.")
        return

    latest = max(versions, key=lambda v: int(v.version))
    version_num = latest.version

    if auc_roc >= AUC_PROMOTION_THRESHOLD:
        # Archive the current Production model (if one exists)
        try:
            prod_versions = client.get_latest_versions(
                REGISTERED_MODEL_NAME, stages=["Production"]
            )
            for old_prod in prod_versions:
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=old_prod.version,
                    stage="Archived",
                )
                print(f"  📦 Version {old_prod.version} archived")
        except Exception:
            pass  # no existing Production model, that's fine

        # Promote new version to Production
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version_num,
            stage="Production",
        )
        print(f"  🚀 Version {version_num} promoted to PRODUCTION")
        print(f"     API can now load via: models:/{REGISTERED_MODEL_NAME}/Production")

    else:
        # Not good enough for Production — put in Staging for review
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version_num,
            stage="Staging",
        )
        print(f"  ⏸️  Version {version_num} moved to STAGING")
        print(f"     AUC {auc_roc} < threshold {AUC_PROMOTION_THRESHOLD}")
        print(f"     Manual review required before promoting to Production.")

    # Tag the version with metadata for auditability
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version_num,
        key="auc_roc",
        value=str(auc_roc),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version_num,
        key="run_id",
        value=run_id,
    )


def save_model_local(model, feature_names: list):
    """
    Also save .joblib file locally as fallback.
    Used by: local dev without MLflow server, explainer.py direct calls.
    In production (docker-compose), api.py loads from MLflow registry instead.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    with open(FEATURE_LIST_OUTPUT, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"\n💾 Local fallback saved → {MODEL_OUTPUT}")


# ──────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def run_training_pipeline():
    print("=" * 60)
    print("  FRAUD DETECTION — TRAINING PIPELINE (MLflow)")
    print("=" * 60)

    # 1. Load + engineer
    df = load_data()
    df = engineer_features(df)
    X, y = prepare_features(df)

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Train
    model = train_model(X_train, y_train, X_val, y_val)

    # 4. Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # 5. Save local .joblib (fallback)
    save_model_local(model, list(X.columns))

    # 6. Log to MLflow + register
    print("\n📊 Logging to MLflow...")
    run_id = log_to_mlflow(model, metrics, list(X.columns), X_val, y_val)

    # 7. Promote to Production if AUC passes threshold
    print("\n🏷️  Checking promotion criteria...")
    promote_model_to_production(run_id, metrics["auc_roc"])

    print("\n" + "=" * 60)
    print("  ✅ TRAINING + MLFLOW COMPLETE")
    print(f"  AUC-ROC:  {metrics['auc_roc']}")
    print(f"  Run ID:   {run_id}")
    print(f"  View UI:  mlflow ui  (then open http://localhost:5000)")
    print("=" * 60)

    return model, metrics, run_id


if __name__ == "__main__":
    run_training_pipeline()