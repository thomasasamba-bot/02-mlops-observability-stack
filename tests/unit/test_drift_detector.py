"""
tests/unit/test_drift_detector.py

Unit tests for app/pipeline/drift_detector.py

Tests are structured in four groups:
  1. PSI computation — continuous and discrete features, stable vs drifted
  2. KS-test baseline reconstruction
  3. Full drift check — report structure, severity classification, thresholds
  4. DriftReport — summary, overall_drift flag, Prometheus gauge updates

All tests are fully offline — no MLflow server, no inference server,
no running services required. Schema and data are generated in fixtures.

Usage:
  pytest tests/unit/test_drift_detector.py -v
  pytest tests/unit/test_drift_detector.py -v --tb=short
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def baseline_df(rng) -> pd.DataFrame:
    """500-row baseline dataset drawn from stable distributions."""
    from scripts.data.generate_data import generate_baseline
    return generate_baseline(500, rng)


@pytest.fixture(scope="module")
def drifted_df(rng) -> pd.DataFrame:
    """200-row drifted dataset with shifted distributions."""
    from scripts.data.generate_data import generate_drifted
    return generate_drifted(200, rng)


@pytest.fixture(scope="module")
def schema(baseline_df, tmp_path_factory) -> dict:
    """Feature schema built from the baseline dataset."""
    from app.pipeline.train import build_feature_schema
    return build_feature_schema(baseline_df)


@pytest.fixture(scope="module")
def schema_path(schema, tmp_path_factory) -> Path:
    """Write schema to disk and return the path."""
    tmp  = tmp_path_factory.mktemp("schema")
    path = tmp / "feature_schema.json"
    path.write_text(json.dumps(schema))
    return path


@pytest.fixture(scope="module")
def baseline_features(baseline_df) -> pd.DataFrame:
    from app.pipeline.drift_detector import FEATURE_COLUMNS
    return baseline_df[FEATURE_COLUMNS].copy()


@pytest.fixture(scope="module")
def drifted_features(drifted_df) -> pd.DataFrame:
    from app.pipeline.drift_detector import FEATURE_COLUMNS
    return drifted_df[FEATURE_COLUMNS].copy()


# ---------------------------------------------------------------------------
# 1. PSI computation
# ---------------------------------------------------------------------------

class TestPSIContinuous:

    def test_stable_data_psi_near_zero(self, baseline_df, schema):
        """PSI of baseline vs itself should be ~0."""
        from app.pipeline.drift_detector import _psi_continuous
        for feat in ["age", "income", "credit_score", "debt_to_income"]:
            stats = schema["features"][feat]
            col   = baseline_df[feat].to_numpy()
            psi   = _psi_continuous(col, stats["psi_bins"])
            assert psi < 0.05, \
                f"Stable PSI for '{feat}' is {psi:.4f}, expected < 0.05"

    def test_drifted_income_psi_significant(self, drifted_df, schema):
        """Income was drifted — PSI should exceed alert threshold."""
        from app.pipeline.drift_detector import _psi_continuous, PSI_ALERT_THRESHOLD
        stats = schema["features"]["income"]
        col   = drifted_df["income"].to_numpy()
        psi   = _psi_continuous(col, stats["psi_bins"])
        assert psi >= PSI_ALERT_THRESHOLD, \
            f"Drifted income PSI {psi:.4f} below alert threshold {PSI_ALERT_THRESHOLD}"

    def test_drifted_dti_psi_significant(self, drifted_df, schema):
        from app.pipeline.drift_detector import _psi_continuous, PSI_ALERT_THRESHOLD
        stats = schema["features"]["debt_to_income"]
        col   = drifted_df["debt_to_income"].to_numpy()
        psi   = _psi_continuous(col, stats["psi_bins"])
        assert psi >= PSI_ALERT_THRESHOLD, \
            f"Drifted debt_to_income PSI {psi:.4f} below alert threshold"

    def test_stable_features_do_not_drift(self, drifted_df, schema):
        """Features with no injected drift should stay below alert threshold."""
        from app.pipeline.drift_detector import _psi_continuous, PSI_ALERT_THRESHOLD
        stable_features = ["age", "loan_amount", "employment_years"]
        for feat in stable_features:
            stats = schema["features"][feat]
            col   = drifted_df[feat].to_numpy()
            psi   = _psi_continuous(col, stats["psi_bins"])
            assert psi < PSI_ALERT_THRESHOLD, \
                f"Stable feature '{feat}' PSI {psi:.4f} exceeded alert threshold"

    def test_psi_non_negative(self, drifted_df, schema):
        from app.pipeline.drift_detector import _psi_continuous
        for feat in ["age", "income", "credit_score"]:
            stats = schema["features"][feat]
            col   = drifted_df[feat].to_numpy()
            psi   = _psi_continuous(col, stats["psi_bins"])
            assert psi >= 0.0, f"PSI for '{feat}' is negative: {psi}"


class TestPSIDiscrete:

    def test_stable_missed_payments_near_zero(self, baseline_df, schema):
        """Discrete PSI of baseline vs itself should be near zero."""
        from app.pipeline.drift_detector import _psi_discrete
        freq_table = schema["features"]["missed_payments"]["freq_table"]
        col        = baseline_df["missed_payments"].to_numpy()
        psi        = _psi_discrete(col, freq_table)
        assert psi < 0.05, \
            f"Stable missed_payments discrete PSI is {psi:.4f}, expected < 0.05"

    def test_drifted_missed_payments_significant(self, drifted_df, schema):
        """Drifted missed_payments (Poisson 0.4→1.1) should exceed alert threshold."""
        from app.pipeline.drift_detector import _psi_discrete, PSI_ALERT_THRESHOLD
        freq_table = schema["features"]["missed_payments"]["freq_table"]
        col        = drifted_df["missed_payments"].to_numpy()
        psi        = _psi_discrete(col, freq_table)
        assert psi >= PSI_ALERT_THRESHOLD, \
            f"Drifted missed_payments PSI {psi:.4f} below alert threshold"

    def test_stable_num_credit_lines_near_zero(self, baseline_df, schema):
        from app.pipeline.drift_detector import _psi_discrete
        freq_table = schema["features"]["num_credit_lines"]["freq_table"]
        col        = baseline_df["num_credit_lines"].to_numpy()
        psi        = _psi_discrete(col, freq_table)
        assert psi < 0.10, \
            f"Stable num_credit_lines discrete PSI is {psi:.4f}, expected < 0.10"

    def test_discrete_psi_non_negative(self, drifted_df, schema):
        from app.pipeline.drift_detector import _psi_discrete
        for feat in ["missed_payments", "num_credit_lines"]:
            freq_table = schema["features"][feat]["freq_table"]
            col        = drifted_df[feat].to_numpy()
            psi        = _psi_discrete(col, freq_table)
            assert psi >= 0.0, f"Discrete PSI for '{feat}' is negative: {psi}"

    def test_unseen_value_handled_gracefully(self, schema):
        """Values not in the training freq_table should not raise."""
        from app.pipeline.drift_detector import _psi_discrete
        freq_table = schema["features"]["missed_payments"]["freq_table"]
        # Include a value far outside training range
        col = np.array([0, 0, 1, 50, 100], dtype=float)
        psi = _psi_discrete(col, freq_table)
        assert psi >= 0.0


# ---------------------------------------------------------------------------
# 2. KS-test baseline reconstruction
# ---------------------------------------------------------------------------

class TestKSBaseline:

    def test_continuous_baseline_sample_length(self, schema):
        from app.pipeline.drift_detector import _ks_baseline_continuous
        stats  = schema["features"]["income"]
        sample = _ks_baseline_continuous(stats, n=500)
        # Should be approximately n (may differ slightly due to per_bin rounding)
        assert 400 <= len(sample) <= 600

    def test_continuous_baseline_in_range(self, schema):
        from app.pipeline.drift_detector import _ks_baseline_continuous
        for feat in ["income", "credit_score", "debt_to_income"]:
            stats  = schema["features"][feat]
            sample = _ks_baseline_continuous(stats)
            assert sample.min() >= stats["min"] * 0.9, \
                f"Baseline sample for '{feat}' below training min"
            assert sample.max() <= stats["max"] * 1.1, \
                f"Baseline sample for '{feat}' above training max"

    def test_discrete_baseline_sample_values(self, schema):
        from app.pipeline.drift_detector import _ks_baseline_discrete
        freq_table = schema["features"]["missed_payments"]["freq_table"]
        sample     = _ks_baseline_discrete(freq_table, n=1000)
        valid_vals = set(int(k) for k in freq_table.keys())
        assert set(sample.astype(int)).issubset(valid_vals), \
            "Discrete baseline contains values not in freq_table"

    def test_discrete_baseline_proportions_match_freq_table(self, schema):
        """Reconstructed sample should roughly match the training proportions."""
        from app.pipeline.drift_detector import _ks_baseline_discrete
        freq_table = schema["features"]["missed_payments"]["freq_table"]
        sample     = _ks_baseline_discrete(freq_table, n=5000)
        for val_str, expected_prop in freq_table.items():
            val           = int(val_str)
            actual_prop   = float(np.mean(sample == val))
            assert abs(actual_prop - expected_prop) < 0.05, \
                f"Value {val}: expected {expected_prop:.3f}, got {actual_prop:.3f}"


# ---------------------------------------------------------------------------
# 3. Full drift check
# ---------------------------------------------------------------------------

class TestDriftCheck:

    def test_report_has_correct_structure(self, baseline_features, schema):
        from app.pipeline.drift_detector import check_drift, FEATURE_COLUMNS
        report = check_drift(baseline_features, schema)
        assert hasattr(report, "timestamp")
        assert hasattr(report, "window_size")
        assert hasattr(report, "feature_results")
        assert hasattr(report, "drifted_features")
        assert hasattr(report, "overall_drift")
        assert len(report.feature_results) == len(FEATURE_COLUMNS)

    def test_stable_data_no_overall_drift(self, baseline_features, schema):
        """Baseline vs itself should not trigger overall drift alert."""
        from app.pipeline.drift_detector import check_drift
        report = check_drift(baseline_features, schema)
        assert not report.overall_drift, \
            f"False positive drift on stable data: {report.drifted_features}"

    def test_drifted_data_triggers_drift(self, drifted_features, schema):
        """Drifted dataset should trigger overall_drift=True."""
        from app.pipeline.drift_detector import check_drift
        report = check_drift(drifted_features, schema)
        assert report.overall_drift, \
            "Drifted dataset did not trigger overall_drift=True"

    def test_drifted_features_are_correct(self, drifted_features, schema):
        """Injected drift features should appear in drifted_features list."""
        from app.pipeline.drift_detector import check_drift
        report = check_drift(drifted_features, schema)
        # These features had drift injected
        expected_drifted = {"income", "debt_to_income", "missed_payments"}
        detected         = set(report.drifted_features)
        overlap          = expected_drifted & detected
        assert len(overlap) >= 2, \
            f"Expected at least 2 of {expected_drifted} in drifted_features, got {detected}"

    def test_stable_features_not_in_drifted_list(self, drifted_features, schema):
        from app.pipeline.drift_detector import check_drift
        report = check_drift(drifted_features, schema)
        # These features had NO drift injected
        stable = {"age", "loan_amount", "employment_years", "num_credit_lines"}
        false_positives = stable & set(report.drifted_features)
        assert not false_positives, \
            f"False positive drift detected for stable features: {false_positives}"

    def test_feature_results_severity_classification(self, drifted_features, schema):
        """
        Severity mirrors the detector logic exactly:
          PSI >= ALERT                          → "alert"
          PSI >= WARN  OR  KS p-value < alpha   → "warning"
          otherwise                             → "stable"
        The KS-test can independently trigger "warning" even when PSI < WARN_THRESHOLD,
        so we validate against both conditions, not PSI alone.
        """
        from app.pipeline.drift_detector import (
            check_drift, PSI_WARN_THRESHOLD, PSI_ALERT_THRESHOLD, KS_ALPHA,
        )
        report = check_drift(drifted_features, schema)
        for r in report.feature_results:
            if r.psi >= PSI_ALERT_THRESHOLD:
                assert r.severity == "alert", \
                    f"Feature '{r.feature}' PSI={r.psi:.4f} should be 'alert'"
            elif r.psi >= PSI_WARN_THRESHOLD or r.ks_pvalue < KS_ALPHA:
                assert r.severity in ("warning", "alert"), \
                    f"Feature '{r.feature}' PSI={r.psi:.4f} KS-p={r.ks_pvalue:.4f} should be 'warning'"
            else:
                assert r.severity == "stable", \
                    f"Feature '{r.feature}' PSI={r.psi:.4f} KS-p={r.ks_pvalue:.4f} should be 'stable'"

    def test_window_too_small_returns_error_report(self, schema):
        from app.pipeline.drift_detector import check_drift, MIN_WINDOW_SIZE
        tiny_df = pd.DataFrame(
            [[35, 50000, 15000, 650, 0.35, 5.0, 4, 0]] * (MIN_WINDOW_SIZE - 1),
            columns=["age", "income", "loan_amount", "credit_score",
                     "debt_to_income", "employment_years", "num_credit_lines",
                     "missed_payments"],
        )
        report = check_drift(tiny_df, schema)
        assert report.error is not None
        assert not report.overall_drift

    def test_report_window_size_matches_input(self, drifted_features, schema):
        from app.pipeline.drift_detector import check_drift
        report = check_drift(drifted_features, schema)
        assert report.window_size == len(drifted_features)

    def test_all_psi_values_non_negative(self, drifted_features, schema):
        from app.pipeline.drift_detector import check_drift
        report = check_drift(drifted_features, schema)
        for r in report.feature_results:
            assert r.psi >= 0.0, \
                f"Feature '{r.feature}' has negative PSI: {r.psi}"

    def test_ks_pvalues_in_unit_interval(self, drifted_features, schema):
        from app.pipeline.drift_detector import check_drift
        report = check_drift(drifted_features, schema)
        for r in report.feature_results:
            assert 0.0 <= r.ks_pvalue <= 1.0, \
                f"Feature '{r.feature}' KS p-value {r.ks_pvalue} outside [0, 1]"


# ---------------------------------------------------------------------------
# 4. Batch report (standalone entry point)
# ---------------------------------------------------------------------------

class TestBatchReport:

    def test_batch_report_on_drifted_csv(self, schema_path, tmp_path_factory):
        """run_batch_report should detect drift in credit_drifted.csv."""
        drifted_csv = Path("data/raw/credit_drifted.csv")
        if not drifted_csv.exists():
            pytest.skip("data/raw/credit_drifted.csv not found — run generate_data.py")

        from app.pipeline.drift_detector import run_batch_report
        report = run_batch_report(drifted_csv, schema_path)
        assert report.overall_drift, \
            "Batch report did not detect drift in credit_drifted.csv"

    def test_batch_report_on_baseline_csv(self, schema_path):
        """run_batch_report should NOT detect drift in credit_baseline.csv."""
        baseline_csv = Path("data/raw/credit_baseline.csv")
        if not baseline_csv.exists():
            pytest.skip("data/raw/credit_baseline.csv not found — run generate_data.py")

        from app.pipeline.drift_detector import run_batch_report
        report = run_batch_report(baseline_csv, schema_path)
        assert not report.overall_drift, \
            f"False positive drift on baseline CSV: {report.drifted_features}"

    def test_batch_report_raises_on_missing_columns(self, schema_path, tmp_path):
        from app.pipeline.drift_detector import run_batch_report
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing"):
            run_batch_report(bad_csv, schema_path)

    def test_batch_report_window_sampling(self, schema_path):
        """window parameter should limit rows used."""
        drifted_csv = Path("data/raw/credit_drifted.csv")
        if not drifted_csv.exists():
            pytest.skip("data/raw/credit_drifted.csv not found")

        from app.pipeline.drift_detector import run_batch_report
        report = run_batch_report(drifted_csv, schema_path, window=100)
        assert report.window_size == 100


# ---------------------------------------------------------------------------
# 5. Confidence metrics
# ---------------------------------------------------------------------------

class TestConfidenceMetrics:

    def test_returns_correct_tuple(self):
        from app.pipeline.drift_detector import update_confidence_metrics
        probs     = [0.1, 0.3, 0.7, 0.9]
        decisions = [0, 0, 1, 1]
        mean, std, dr = update_confidence_metrics(probs, decisions)
        assert abs(mean - np.mean(probs)) < 1e-6
        assert abs(std  - np.std(probs))  < 1e-6
        assert abs(dr   - 0.5)            < 1e-6

    def test_empty_input_returns_zeros(self):
        from app.pipeline.drift_detector import update_confidence_metrics
        mean, std, dr = update_confidence_metrics([], [])
        assert mean == 0.0
        assert std  == 0.0
        assert dr   == 0.0

    def test_all_defaults(self):
        from app.pipeline.drift_detector import update_confidence_metrics
        mean, std, dr = update_confidence_metrics(
            [0.8, 0.9, 0.7], [1, 1, 1]
        )
        assert dr == 1.0

    def test_no_defaults(self):
        from app.pipeline.drift_detector import update_confidence_metrics
        mean, std, dr = update_confidence_metrics(
            [0.1, 0.2, 0.15], [0, 0, 0]
        )
        assert dr == 0.0