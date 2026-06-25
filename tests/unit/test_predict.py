"""
tests/unit/test_predict.py

Unit tests for app/pipeline/predict.py

Tests are structured in four groups:
  1. Input validation — missing fields, bad types, extra keys
  2. Single prediction — probability range, decision threshold, confidence flag
  3. Batch prediction — ordering, batch vs single consistency, empty batch
  4. Prediction buffer — rolling buffer population, stats, thread safety

These tests are fully offline: the `loaded_cache` fixture trains a tiny
throwaway model and registers it into a temporary, file-backed SQLite
MLflow store (created fresh per test session), then points predict.py's
model cache at it directly. No live MLflow server or pre-existing
registered model is required — this mirrors the pattern already used by
tests/unit/test_training.py.

(This previously hard-coded a check against http://localhost:5000 and
skipped 22 of these 30 tests whenever that server wasn't already running
with a model registered — which is exactly the case in CI, since
.github/workflows/ci.yml's unit-tests job never starts an MLflow server.
Those tests were silently SKIPPED, not "passed", despite this file being
labeled fully offline.)

Usage:
  pytest tests/unit/test_predict.py -v
  pytest tests/unit/test_predict.py -v -k "not buffer"   # skip buffer tests
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOW_RISK = {
    "age": 50, "income": 120_000, "loan_amount": 10_000,
    "credit_score": 800, "debt_to_income": 0.10,
    "employment_years": 20.0, "num_credit_lines": 8,
    "missed_payments": 0,
}

HIGH_RISK = {
    "age": 24, "income": 22_000, "loan_amount": 20_000,
    "credit_score": 490, "debt_to_income": 0.80,
    "employment_years": 0.3, "num_credit_lines": 1,
    "missed_payments": 5,
}

BORDERLINE = {
    "age": 35, "income": 48_000, "loan_amount": 15_000,
    "credit_score": 620, "debt_to_income": 0.42,
    "employment_years": 3.0, "num_credit_lines": 4,
    "missed_payments": 1,
}

FEATURE_COLUMNS = [
    "age", "income", "loan_amount", "credit_score",
    "debt_to_income", "employment_years", "num_credit_lines", "missed_payments",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_cache(tmp_path_factory):
    """
    Train a tiny throwaway RandomForest on synthetic data, register it into
    a temp SQLite-backed MLflow store under the same name/stage predict.py
    looks for, then point predict.py's module-level singleton model cache
    (`_cache`) at that store and force-load it.

    predict.py reads MLFLOW_TRACKING_URI / MODEL_STAGE into module-level
    constants once, at import time — setting the env var alone has no
    effect if the module was already imported elsewhere first. We patch
    the already-imported module's constants directly so this is correct
    regardless of test collection/import order.

    NOTE: this fixture is module-scoped, but pytest's built-in
    `monkeypatch` fixture is function-scoped — depending on it here would
    raise ScopeMismatch. We use `pytest.MonkeyPatch()` directly instead,
    which carries no scope restriction, and undo it ourselves at teardown.
    """
    # Reuse the project's own synthetic data generator (same one
    # test_training.py uses) so the model actually learns a real
    # credit-risk → default relationship. Pure random/uncorrelated
    # labels would make assertions like "HIGH_RISK probability >
    # LOW_RISK probability" flaky or simply wrong.
    import sys
    from pathlib import Path as _Path

    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    sys.path.insert(0, str(_Path(__file__).parents[2]))
    from scripts.data.generate_data import generate_baseline

    tmp_dir       = tmp_path_factory.mktemp("mlflow_predict_test")
    tracking_uri  = f"sqlite:///{tmp_dir / 'mlflow.db'}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    rng = np.random.default_rng(7)
    train_df = generate_baseline(1500, rng)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["default"]
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=7,
    ).fit(X, y)

    model_name = "credit-scoring-model"
    experiment = mlflow.set_experiment("predict-unit-tests")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.sklearn.log_model(
            clf, artifact_path="model",
            registered_model_name=model_name,
        )

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    mv = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=model_name, version=mv.version,
        stage="Staging", archive_existing_versions=False,
    )

    import app.pipeline.predict as predict_mod

    mp = pytest.MonkeyPatch()
    mp.setattr(predict_mod, "MLFLOW_URI", tracking_uri, raising=True)
    mp.setattr(predict_mod, "MODEL_NAME", model_name, raising=True)
    mp.setattr(predict_mod, "MODEL_STAGE", "Staging", raising=True)
    mp.setattr(predict_mod, "MODEL_ALIAS", None, raising=True)

    cache = predict_mod._cache
    # Force a clean reload against the patched constants even if some
    # earlier-collected test already populated the cache from elsewhere.
    with cache._lock:
        cache._model = None
        cache._last_check = 0.0
    cache.ensure_loaded()

    if cache.model is None:
        mp.undo()
        pytest.skip("Model could not be loaded into the test MLflow store")

    yield cache

    mp.undo()


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_input_returns_dataframe(self):
        import pandas as pd

        from app.pipeline.predict import validate_features
        df = validate_features(LOW_RISK)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == FEATURE_COLUMNS
        assert len(df) == 1

    def test_missing_single_feature_raises(self):
        from app.pipeline.predict import ValidationError, validate_features
        bad = {k: v for k, v in LOW_RISK.items() if k != "credit_score"}
        with pytest.raises(ValidationError, match="credit_score"):
            validate_features(bad)

    def test_missing_multiple_features_raises(self):
        from app.pipeline.predict import ValidationError, validate_features
        bad = {"age": 35, "income": 50000}
        with pytest.raises(ValidationError):
            validate_features(bad)

    def test_non_numeric_feature_raises(self):
        from app.pipeline.predict import ValidationError, validate_features
        bad = {**LOW_RISK, "credit_score": "excellent"}
        with pytest.raises(ValidationError, match="credit_score"):
            validate_features(bad)

    def test_extra_keys_are_ignored(self):
        from app.pipeline.predict import validate_features
        extra = {**LOW_RISK, "extra_field": 999, "another": "ignored"}
        df = validate_features(extra)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_integer_values_accepted(self):
        """Integer inputs should be coerced to float without error."""
        from app.pipeline.predict import validate_features
        int_input = {k: int(v) for k, v in LOW_RISK.items()}
        df = validate_features(int_input)
        assert df.dtypes.eq(float).all() or df.dtypes.apply(
            lambda t: np.issubdtype(t, np.number)
        ).all()

    def test_empty_batch_raises(self):
        from app.pipeline.predict import ValidationError, validate_batch
        with pytest.raises(ValidationError, match="[Ee]mpty"):
            validate_batch([])

    def test_batch_with_bad_record_raises(self):
        from app.pipeline.predict import ValidationError, validate_batch
        bad_record = {k: v for k, v in LOW_RISK.items() if k != "income"}
        with pytest.raises(ValidationError):
            validate_batch([LOW_RISK, bad_record])


# ---------------------------------------------------------------------------
# 2. Single prediction
# ---------------------------------------------------------------------------

class TestSinglePrediction:

    def test_predict_returns_result(self, loaded_cache):
        from app.pipeline.predict import predict
        result = predict(LOW_RISK)
        assert result is not None

    def test_probability_in_unit_interval(self, loaded_cache):
        from app.pipeline.predict import predict
        for sample in [LOW_RISK, HIGH_RISK, BORDERLINE]:
            result = predict(sample)
            assert 0.0 <= result.probability <= 1.0, \
                f"Probability {result.probability} outside [0, 1]"

    def test_decision_is_binary(self, loaded_cache):
        from app.pipeline.predict import predict
        for sample in [LOW_RISK, HIGH_RISK, BORDERLINE]:
            result = predict(sample)
            assert result.decision in (0, 1), \
                f"Decision {result.decision} is not 0 or 1"

    def test_low_risk_predicts_no_default(self, loaded_cache):
        from app.pipeline.predict import predict
        result = predict(LOW_RISK)
        assert result.decision == 0, \
            f"Low-risk applicant predicted as default (prob={result.probability:.3f})"

    def test_high_risk_predicts_default(self, loaded_cache):
        from app.pipeline.predict import predict
        result = predict(HIGH_RISK)
        assert result.decision == 1, \
            f"High-risk applicant not predicted as default (prob={result.probability:.3f})"

    def test_low_risk_lower_probability_than_high_risk(self, loaded_cache):
        from app.pipeline.predict import predict
        p_low  = predict(LOW_RISK).probability
        p_high = predict(HIGH_RISK).probability
        assert p_low < p_high, \
            f"Low-risk prob {p_low:.3f} should be < high-risk {p_high:.3f}"

    def test_decision_consistent_with_threshold(self, loaded_cache):
        from app.pipeline.predict import DECISION_THRESHOLD, predict
        for sample in [LOW_RISK, HIGH_RISK, BORDERLINE]:
            result = predict(sample)
            expected = 1 if result.probability >= DECISION_THRESHOLD else 0
            assert result.decision == expected, (
                f"Decision {result.decision} inconsistent with "
                f"prob={result.probability:.3f} threshold={DECISION_THRESHOLD}"
            )

    def test_low_confidence_flag_when_probability_in_band(self, loaded_cache):
        from app.pipeline.predict import LOW_CONF_HIGH, LOW_CONF_LOW, predict
        result = predict(BORDERLINE)
        expected_low_conf = LOW_CONF_LOW <= result.probability <= LOW_CONF_HIGH
        assert result.low_confidence == expected_low_conf, (
            f"low_confidence={result.low_confidence} but prob={result.probability:.3f} "
            f"band=[{LOW_CONF_LOW}, {LOW_CONF_HIGH}]"
        )

    def test_model_version_is_string(self, loaded_cache):
        from app.pipeline.predict import predict
        result = predict(LOW_RISK)
        assert isinstance(result.model_version, str)
        assert len(result.model_version) > 0

    def test_latency_is_positive(self, loaded_cache):
        from app.pipeline.predict import predict
        result = predict(LOW_RISK)
        assert result.latency_ms > 0

    def test_features_used_matches_columns(self, loaded_cache):
        from app.pipeline.predict import predict
        result = predict(LOW_RISK)
        assert result.features_used == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 3. Batch prediction
# ---------------------------------------------------------------------------

class TestBatchPrediction:

    def test_batch_returns_correct_count(self, loaded_cache):
        from app.pipeline.predict import predict_batch
        samples = [LOW_RISK, HIGH_RISK, BORDERLINE]
        results = predict_batch(samples)
        assert len(results) == 3

    def test_batch_preserves_order(self, loaded_cache):
        """Result[i] should correspond to input[i]."""
        from app.pipeline.predict import predict, predict_batch
        samples = [LOW_RISK, HIGH_RISK, BORDERLINE]
        batch   = predict_batch(samples)
        singles = [predict(s) for s in samples]
        for i, (b, s) in enumerate(zip(batch, singles)):
            assert abs(b.probability - s.probability) < 1e-9, \
                f"Record {i}: batch prob {b.probability:.6f} != single {s.probability:.6f}"

    def test_batch_of_one(self, loaded_cache):
        from app.pipeline.predict import predict_batch
        results = predict_batch([LOW_RISK])
        assert len(results) == 1

    def test_batch_large(self, loaded_cache):
        from app.pipeline.predict import predict_batch
        samples = [LOW_RISK] * 50 + [HIGH_RISK] * 50
        results = predict_batch(samples)
        assert len(results) == 100
        low_probs  = [r.probability for r in results[:50]]
        high_probs = [r.probability for r in results[50:]]
        assert np.mean(low_probs) < np.mean(high_probs)

    def test_batch_all_probabilities_valid(self, loaded_cache):
        from app.pipeline.predict import predict_batch
        samples = [LOW_RISK, HIGH_RISK, BORDERLINE] * 10
        results = predict_batch(samples)
        for r in results:
            assert 0.0 <= r.probability <= 1.0


# ---------------------------------------------------------------------------
# 4. Prediction buffer
# ---------------------------------------------------------------------------

class TestPredictionBuffer:

    def test_predictions_appear_in_buffer(self, loaded_cache):
        from app.pipeline.predict import _prediction_buffer, get_recent_predictions, predict
        initial_size = len(_prediction_buffer)
        predict(LOW_RISK)
        predict(HIGH_RISK)
        records = get_recent_predictions()
        assert len(records) >= initial_size + 2

    def test_buffer_records_have_correct_fields(self, loaded_cache):
        from app.pipeline.predict import get_recent_predictions, predict
        predict(LOW_RISK)
        records = get_recent_predictions(1)
        assert len(records) == 1
        r = records[0]
        assert hasattr(r, "probability")
        assert hasattr(r, "decision")
        assert hasattr(r, "features")
        assert hasattr(r, "timestamp")
        assert hasattr(r, "model_version")

    def test_get_recent_predictions_limit(self, loaded_cache):
        from app.pipeline.predict import get_recent_predictions, predict_batch
        predict_batch([LOW_RISK] * 20)
        records = get_recent_predictions(5)
        assert len(records) == 5

    def test_prediction_stats_structure(self, loaded_cache):
        from app.pipeline.predict import get_prediction_stats, predict
        predict(LOW_RISK)
        stats = get_prediction_stats()
        assert "buffered_predictions" in stats
        assert "model_version"        in stats
        assert "confidence"            in stats
        assert "default_rate"         in stats
        conf = stats["confidence"]
        assert "mean" in conf
        assert "std"  in conf
        assert "p50"  in conf

    def test_prediction_stats_confidence_in_range(self, loaded_cache):
        from app.pipeline.predict import get_prediction_stats, predict_batch
        predict_batch([LOW_RISK, HIGH_RISK] * 10)
        stats = get_prediction_stats()
        mean = stats["confidence"]["mean"]
        assert 0.0 <= mean <= 1.0, f"Confidence mean {mean} outside [0, 1]"

    def test_buffer_thread_safety(self, loaded_cache):
        """Concurrent predictions from multiple threads should not corrupt the buffer."""
        from app.pipeline.predict import get_recent_predictions, predict

        errors = []

        def worker():
            try:
                for _ in range(10):
                    predict(LOW_RISK)
                    predict(HIGH_RISK)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        records = get_recent_predictions()
        # All records should have valid probabilities
        for r in records:
            assert 0.0 <= r.probability <= 1.0