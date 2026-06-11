"""
tests/unit/test_training.py

Unit tests for app/pipeline/train.py

Tests are structured in three groups:
  1. Data loading and validation
  2. Feature schema generation (including freq_table for discrete features)
  3. Model training quality (CV ROC-AUC, test metrics, feature importances)

These tests run WITHOUT a running MLflow server — they use a local SQLite
tracking URI so they work in CI without any external dependencies.

Usage:
  pytest tests/unit/test_training.py -v
  pytest tests/unit/test_training.py -v --tb=short
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline_csv(tmp_path_factory) -> Path:
    """Generate a small baseline CSV for testing (500 rows)."""
    tmp      = tmp_path_factory.mktemp("data")
    csv_path = tmp / "credit_baseline.csv"

    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from scripts.data.generate_data import generate_baseline

    rng = np.random.default_rng(42)
    df  = generate_baseline(500, rng)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def trained_pipeline(baseline_csv, tmp_path_factory):
    """
    Train a pipeline on the test dataset and return
    (pipeline, X_test, y_test, schema, schema_path).

    Uses SQLite MLflow tracking — no server required.
    """
    import mlflow
    import mlflow.sklearn
    from app.pipeline.train import (
        FEATURE_COLUMNS, TARGET_COLUMN,
        build_pipeline, build_feature_schema,
        load_data, save_feature_schema,
    )

    tmp        = tmp_path_factory.mktemp("mlflow")
    db_path    = tmp / "mlflow.db"
    schema_path = tmp / "feature_schema.json"

    # SQLite backend works in all MLflow versions including 2.9+
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("test-credit-scoring")

    df = load_data(baseline_csv)
    X  = df[FEATURE_COLUMNS]
    y  = df[TARGET_COLUMN]

    schema = build_feature_schema(df)
    save_feature_schema(schema, schema_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipeline = build_pipeline(n_estimators=50, max_depth=None, random_state=42)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        proba   = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
        mlflow.log_metric("test_roc_auc", roc_auc)

    return pipeline, X_test, y_test, schema, schema_path


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

class TestDataLoading:

    def test_loads_correct_columns(self, baseline_csv):
        from app.pipeline.train import load_data, FEATURE_COLUMNS, TARGET_COLUMN
        df = load_data(baseline_csv)
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            assert col in df.columns, f"Missing column: {col}"

    def test_loads_expected_row_count(self, baseline_csv):
        from app.pipeline.train import load_data
        df = load_data(baseline_csv)
        assert len(df) == 500

    def test_default_rate_in_range(self, baseline_csv):
        from app.pipeline.train import load_data, TARGET_COLUMN
        df = load_data(baseline_csv)
        dr = df[TARGET_COLUMN].mean()
        assert 0.10 <= dr <= 0.40, \
            f"Default rate {dr:.2f} outside expected range [0.10, 0.40]"

    def test_no_missing_values(self, baseline_csv):
        from app.pipeline.train import load_data, FEATURE_COLUMNS
        df    = load_data(baseline_csv)
        nulls = df[FEATURE_COLUMNS].isnull().sum()
        assert nulls.sum() == 0, f"Unexpected nulls:\n{nulls[nulls > 0]}"

    def test_feature_ranges(self, baseline_csv):
        from app.pipeline.train import load_data
        df = load_data(baseline_csv)
        assert df["credit_score"].between(300, 850).all()
        assert df["debt_to_income"].between(0, 1).all()
        assert df["age"].between(18, 80).all()
        assert (df["missed_payments"] >= 0).all()

    def test_raises_on_missing_file(self, tmp_path):
        from app.pipeline.train import load_data
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        from app.pipeline.train import load_data
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing"):
            load_data(bad_csv)


# ---------------------------------------------------------------------------
# 2. Feature schema
# ---------------------------------------------------------------------------

class TestFeatureSchema:

    def test_schema_has_all_features(self, trained_pipeline):
        _, _, _, schema, _ = trained_pipeline
        from app.pipeline.train import FEATURE_COLUMNS
        for col in FEATURE_COLUMNS:
            assert col in schema["features"], f"Missing feature in schema: {col}"

    def test_schema_has_required_keys(self, trained_pipeline):
        _, _, _, schema, _ = trained_pipeline
        required = {"mean", "std", "min", "max", "p25", "p50", "p75", "p95", "psi_bins"}
        for feat, stats in schema["features"].items():
            missing = required - set(stats.keys())
            assert not missing, \
                f"Feature '{feat}' schema missing keys: {missing}"

    def test_psi_bins_length(self, trained_pipeline):
        _, _, _, schema, _ = trained_pipeline
        for feat, stats in schema["features"].items():
            bins = stats["psi_bins"]
            assert len(bins) == 11, \
                f"Feature '{feat}' psi_bins has {len(bins)} entries, expected 11"

    def test_psi_bins_are_sorted(self, trained_pipeline):
        _, _, _, schema, _ = trained_pipeline
        for feat, stats in schema["features"].items():
            bins = stats["psi_bins"]
            assert bins == sorted(bins), \
                f"Feature '{feat}' psi_bins not sorted"

    def test_discrete_features_have_freq_table(self, trained_pipeline):
        _, _, _, schema, _ = trained_pipeline
        for feat in {"num_credit_lines", "missed_payments"}:
            assert "freq_table" in schema["features"][feat], \
                f"Discrete feature '{feat}' missing freq_table"

    def test_freq_table_sums_to_one(self, trained_pipeline):
        _, _, _, schema, _ = trained_pipeline
        for feat in {"num_credit_lines", "missed_payments"}:
            total = sum(schema["features"][feat]["freq_table"].values())
            assert abs(total - 1.0) < 0.01, \
                f"freq_table for '{feat}' sums to {total:.4f}, expected ~1.0"

    def test_schema_target_fields(self, trained_pipeline):
        _, _, _, schema, _ = trained_pipeline
        assert "default_rate" in schema["target"]
        assert "n_samples"    in schema["target"]
        assert 0 < schema["target"]["default_rate"] < 1
        assert schema["target"]["n_samples"] == 500

    def test_schema_serialises_to_json(self, trained_pipeline):
        _, _, _, schema, _ = trained_pipeline
        json_str = json.dumps(schema)
        reloaded = json.loads(json_str)
        assert reloaded["target"]["n_samples"] == 500

    def test_schema_saved_to_disk(self, trained_pipeline):
        _, _, _, _, schema_path = trained_pipeline
        assert schema_path.exists()
        reloaded = json.loads(schema_path.read_text())
        assert "features" in reloaded
        assert "target"   in reloaded


# ---------------------------------------------------------------------------
# 3. Model quality
# ---------------------------------------------------------------------------

class TestModelQuality:

    def test_pipeline_predicts_probabilities(self, trained_pipeline):
        pipeline, X_test, _, _, _ = trained_pipeline
        proba = pipeline.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert (proba >= 0).all() and (proba <= 1).all()
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_probability_spread(self, trained_pipeline):
        """Probabilities should span a wide range — not all clustered near 0.5."""
        pipeline, X_test, _, _, _ = trained_pipeline
        proba = pipeline.predict_proba(X_test)[:, 1]
        spread = proba.max() - proba.min()
        assert spread > 0.3, \
            f"Probability range too narrow: [{proba.min():.3f}, {proba.max():.3f}]"

    def test_roc_auc_above_threshold(self, trained_pipeline):
        """ROC-AUC should be meaningfully above chance (0.70 minimum)."""
        pipeline, X_test, y_test, _, _ = trained_pipeline
        proba   = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
        assert roc_auc >= 0.70, \
            f"ROC-AUC {roc_auc:.4f} below minimum threshold 0.70"

    def test_predictions_not_all_same_class(self, trained_pipeline):
        """Model should predict both classes."""
        pipeline, X_test, _, _, _ = trained_pipeline
        preds = pipeline.predict(X_test)
        assert len(set(preds)) > 1, "Model predicts only one class"

    def test_feature_importances_sum_to_one(self, trained_pipeline):
        pipeline, _, _, _, _ = trained_pipeline
        importances = pipeline.named_steps["clf"].feature_importances_
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_top_feature_is_expected(self, trained_pipeline):
        """Top feature should be one of the known strong predictors."""
        from app.pipeline.train import FEATURE_COLUMNS
        pipeline, _, _, _, _ = trained_pipeline
        importances = dict(zip(
            FEATURE_COLUMNS,
            pipeline.named_steps["clf"].feature_importances_,
        ))
        top = max(importances, key=importances.get)
        expected = {"credit_score", "income", "missed_payments",
                    "debt_to_income", "loan_amount"}
        assert top in expected, f"Unexpected top feature: {top}"

    def test_pipeline_has_scaler_and_clf(self, trained_pipeline):
        pipeline, _, _, _, _ = trained_pipeline
        assert "scaler" in pipeline.named_steps
        assert "clf"    in pipeline.named_steps

    def test_low_risk_scores_lower_than_high_risk(self, trained_pipeline):
        """Sanity check: textbook low-risk applicant scores below high-risk."""
        from app.pipeline.train import FEATURE_COLUMNS
        pipeline, _, _, _, _ = trained_pipeline

        low_risk = pd.DataFrame([{
            "age": 50, "income": 120000, "loan_amount": 10000,
            "credit_score": 800, "debt_to_income": 0.10,
            "employment_years": 20.0, "num_credit_lines": 8,
            "missed_payments": 0,
        }], columns=FEATURE_COLUMNS)

        high_risk = pd.DataFrame([{
            "age": 24, "income": 22000, "loan_amount": 20000,
            "credit_score": 490, "debt_to_income": 0.80,
            "employment_years": 0.3, "num_credit_lines": 1,
            "missed_payments": 5,
        }], columns=FEATURE_COLUMNS)

        p_low  = pipeline.predict_proba(low_risk)[0, 1]
        p_high = pipeline.predict_proba(high_risk)[0, 1]
        assert p_low < p_high, \
            f"Low-risk prob {p_low:.3f} should be < high-risk {p_high:.3f}"