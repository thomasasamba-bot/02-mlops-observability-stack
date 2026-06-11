"""
app/pipeline/train.py

Credit scoring model training pipeline with full MLflow experiment tracking.

What this does:
  1. Loads baseline CSV (produced by scripts/data/generate_data.py)
  2. Preprocesses features (scaling + encoding)
  3. Trains a RandomForestClassifier with cross-validation
  4. Logs params, metrics, feature importances, and the model to MLflow
  5. Registers the model in the MLflow Model Registry as "credit-scoring-model"
  6. Saves a feature schema artifact so predict.py and drift_detector.py
     can validate incoming data against training distributions

Outputs:
  - MLflow run with full metrics + artifacts
  - Registered model version (Staging stage)
  - data/processed/feature_schema.json  (column stats for drift baseline)

Usage:
  python -m app.pipeline.train
  python -m app.pipeline.train --data-path data/raw/credit_baseline.csv
  python -m app.pipeline.train --data-path data/raw/credit_baseline.csv \\
      --experiment-name credit-scoring --n-estimators 200 --run-name v2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "num_credit_lines",
    "missed_payments",
]
TARGET_COLUMN = "default"
MODEL_NAME = "credit-scoring-model"
DEFAULT_EXPERIMENT = "credit-scoring"
DEFAULT_DATA_PATH = Path("data/raw/credit_baseline.csv")
SCHEMA_OUTPUT_PATH = Path("data/processed/feature_schema.json")


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    log.info("Loading data from %s", data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Run: python scripts/data/generate_data.py"
        )
    df = pd.read_csv(data_path)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    log.info(
        "Loaded %d rows, %d cols  |  default_rate=%.1f%%",
        len(df), len(df.columns), df[TARGET_COLUMN].mean() * 100,
    )
    return df


# ---------------------------------------------------------------------------
# Feature schema — saved so drift_detector.py can compare incoming batches
# ---------------------------------------------------------------------------

DISCRETE_FEATURES = {"num_credit_lines", "missed_payments"}


def build_feature_schema(df: pd.DataFrame) -> dict:
    """
    Capture per-feature statistics from the training set.
    drift_detector.py uses this as the reference distribution.

    For discrete/integer features (num_credit_lines, missed_payments) we also
    store a freq_table — the empirical P(X=k) from training data. This is used
    by drift_detector.py for frequency-table PSI, which is correct for sparse
    Poisson-distributed features where percentile bins degenerate.
    """
    schema: dict = {"features": {}, "target": {}}

    for col in FEATURE_COLUMNS:
        series = df[col]
        entry = {
            "mean":   float(series.mean()),
            "std":    float(series.std()),
            "min":    float(series.min()),
            "max":    float(series.max()),
            "p25":    float(series.quantile(0.25)),
            "p50":    float(series.quantile(0.50)),
            "p75":    float(series.quantile(0.75)),
            "p95":    float(series.quantile(0.95)),
            "psi_bins": [
                float(v) for v in np.percentile(series, np.linspace(0, 100, 11))
            ],
        }
        if col in DISCRETE_FEATURES:
            counts = series.value_counts(normalize=True).sort_index()
            entry["freq_table"] = {str(int(k)): float(v) for k, v in counts.items()}

        schema["features"][col] = entry

    schema["target"]["default_rate"] = float(df[TARGET_COLUMN].mean())
    schema["target"]["n_samples"]    = int(len(df))
    return schema


def save_feature_schema(schema: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2))
    log.info("Feature schema saved to %s", path)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_pipeline(n_estimators: int, max_depth: int | None, random_state: int) -> Pipeline:
    """
    StandardScaler + RandomForestClassifier wrapped in a sklearn Pipeline.
    The scaler handles income/loan_amount magnitude differences.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            class_weight="balanced",    # handles ~20% positive rate
            n_jobs=-1,
            random_state=random_state,
        )),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, split: str) -> dict[str, float]:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        f"{split}_accuracy":          float(accuracy_score(y, y_pred)),
        f"{split}_f1":                float(f1_score(y, y_pred, zero_division=0)),
        f"{split}_precision":         float(precision_score(y, y_pred, zero_division=0)),
        f"{split}_recall":            float(recall_score(y, y_pred, zero_division=0)),
        f"{split}_roc_auc":           float(roc_auc_score(y, y_proba)),
        f"{split}_avg_precision":     float(average_precision_score(y, y_proba)),
    }

    log.info(
        "%s  acc=%.4f  f1=%.4f  roc_auc=%.4f  precision=%.4f  recall=%.4f",
        split.upper().ljust(5),
        metrics[f"{split}_accuracy"],
        metrics[f"{split}_f1"],
        metrics[f"{split}_roc_auc"],
        metrics[f"{split}_precision"],
        metrics[f"{split}_recall"],
    )
    return metrics


def cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "f1":      "f1",
        "accuracy": "accuracy",
    }
    cv_metrics: dict[str, float] = {}
    for metric_name, scorer in scoring.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        cv_metrics[f"cv_{metric_name}_mean"] = float(scores.mean())
        cv_metrics[f"cv_{metric_name}_std"]  = float(scores.std())
        log.info(
            "CV %s: %.4f ± %.4f",
            metric_name.ljust(10),
            scores.mean(),
            scores.std(),
        )
    return cv_metrics


# ---------------------------------------------------------------------------
# Feature importance logging
# ---------------------------------------------------------------------------

def log_feature_importances(pipeline: Pipeline, feature_names: list[str]) -> dict[str, float]:
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    importance_dict = dict(zip(feature_names, importances.tolist()))

    log.info("Feature importances:")
    for feat, imp in sorted(importance_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        log.info("  %-22s  %.4f  %s", feat, imp, bar)

    return importance_dict


# ---------------------------------------------------------------------------
# MLflow model registration
# ---------------------------------------------------------------------------

def register_model(run_id: str, model_uri: str) -> None:
    client = mlflow.tracking.MlflowClient()

    # Create registered model if it doesn't exist yet
    try:
        client.create_registered_model(
            name=MODEL_NAME,
            description=(
                "RandomForest credit default classifier. "
                "Trained on synthetic credit scoring dataset. "
                "Observed via PSI + KS-test drift detection."
            ),
        )
        log.info("Created new registered model: %s", MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        log.info("Registered model already exists: %s", MODEL_NAME)

    # Register this run's model as a new version
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    # Transition to Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    log.info(
        "Registered model version %s → Staging  (run_id=%s)",
        model_version.version,
        run_id,
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    data_path: Path,
    experiment_name: str,
    run_name: str | None,
    n_estimators: int,
    max_depth: int | None,
    test_size: float,
    cv_folds: int,
    random_state: int,
    register: bool,
) -> str:
    """
    Full training pipeline. Returns the MLflow run_id.
    """
    # ── Load ──────────────────────────────────────────────────────────────
    df = load_data(data_path)
    X  = df[FEATURE_COLUMNS]
    y  = df[TARGET_COLUMN]

    # ── Feature schema ────────────────────────────────────────────────────
    schema = build_feature_schema(df)
    save_feature_schema(schema, SCHEMA_OUTPUT_PATH)

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    log.info(
        "Split: train=%d  test=%d  (test_size=%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )

    # ── MLflow ────────────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    log.info("MLflow tracking URI: %s  experiment: %s", tracking_uri, experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log.info("MLflow run started: %s", run_id)

        # ── Params ────────────────────────────────────────────────────────
        params = {
            "model_type":    "RandomForestClassifier",
            "n_estimators":  n_estimators,
            "max_depth":     max_depth if max_depth else "unlimited",
            "min_samples_leaf": 5,
            "class_weight":  "balanced",
            "test_size":     test_size,
            "cv_folds":      cv_folds,
            "random_state":  random_state,
            "n_features":    len(FEATURE_COLUMNS),
            "n_train":       len(X_train),
            "n_test":        len(X_test),
            "baseline_default_rate": round(float(y.mean()), 4),
        }
        mlflow.log_params(params)

        # ── Cross-validation ──────────────────────────────────────────────
        log.info("Running %d-fold cross-validation …", cv_folds)
        pipeline = build_pipeline(n_estimators, max_depth, random_state)
        cv_metrics = cross_validate(pipeline, X_train, y_train, cv_folds, random_state)
        mlflow.log_metrics(cv_metrics)

        # ── Final fit ─────────────────────────────────────────────────────
        log.info("Fitting final model on full training set …")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_duration = time.time() - t0
        mlflow.log_metric("train_duration_seconds", round(train_duration, 2))
        log.info("Training complete in %.2fs", train_duration)

        # ── Evaluate ──────────────────────────────────────────────────────
        train_metrics = evaluate(pipeline, X_train, y_train, "train")
        test_metrics  = evaluate(pipeline, X_test,  y_test,  "test")
        mlflow.log_metrics({**train_metrics, **test_metrics})

        # ── Feature importances ───────────────────────────────────────────
        importances = log_feature_importances(pipeline, FEATURE_COLUMNS)
        mlflow.log_metrics({
            f"importance_{k}": v for k, v in importances.items()
        })

        # ── Artifacts ─────────────────────────────────────────────────────
        mlflow.log_artifact(str(SCHEMA_OUTPUT_PATH), artifact_path="schema")
        mlflow.log_artifact(str(data_path), artifact_path="training_data")

        # ── Log model ─────────────────────────────────────────────────────
        signature  = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=MODEL_NAME if register else None,
        )

        log.info("Model logged: %s", model_info.model_uri)

        # ── Register in Model Registry ────────────────────────────────────
        if register:
            try:
                register_model(run_id, model_info.model_uri)
            except Exception as exc:
                log.warning("Model registration skipped (MLflow registry unavailable): %s", exc)

        # ── Summary tag ───────────────────────────────────────────────────
        mlflow.set_tags({
            "model_name":  MODEL_NAME,
            "data_source": str(data_path),
            "framework":   "scikit-learn",
            "stage":       "training",
        })

        log.info(
            "Run complete  run_id=%s  test_roc_auc=%.4f  test_f1=%.4f",
            run_id,
            test_metrics["test_roc_auc"],
            test_metrics["test_f1"],
        )

    return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train credit scoring RandomForest with MLflow tracking."
    )
    parser.add_argument("--data-path",        type=Path,  default=DEFAULT_DATA_PATH)
    parser.add_argument("--experiment-name",  type=str,   default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name",         type=str,   default=None)
    parser.add_argument("--n-estimators",     type=int,   default=150)
    parser.add_argument("--max-depth",        type=int,   default=None,
                        help="Max tree depth (default: unlimited)")
    parser.add_argument("--test-size",        type=float, default=0.20)
    parser.add_argument("--cv-folds",         type=int,   default=5)
    parser.add_argument("--random-state",     type=int,   default=42)
    parser.add_argument("--no-register",      dest="register", action="store_false",
                        default=True,
                        help="Skip MLflow Model Registry registration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        register=args.register,
    )


if __name__ == "__main__":
    main()