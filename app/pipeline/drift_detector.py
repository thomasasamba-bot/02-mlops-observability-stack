"""
app/pipeline/drift_detector.py

Model-level drift detection for the credit scoring pipeline.

This is SEPARATE from the infrastructure anomaly detection in app/anomaly_detection/.
That layer watches CPU/memory/disk/network. This layer watches what's happening
INSIDE the model: are the features being scored today drawn from the same
distribution the model was trained on?

Two detection methods:
  1. PSI (Population Stability Index)
       - Continuous features: bins are training-set percentile breakpoints,
         so baseline mass = 1/n_bins per bin by construction. Only the
         current distribution needs to be binned.
       - Discrete/integer features (num_credit_lines, missed_payments):
         frequency-table PSI using empirical P(X=k) from the training set.
         Stored in schema as `freq_table`. Avoids the degenerate percentile
         bin problem for sparse Poisson-distributed features.

       PSI < 0.10  → stable
       PSI 0.10–0.25 → moderate drift (warning)
       PSI > 0.25  → significant drift (alert)

  2. KS-test
       Non-parametric distribution shift test. p < 0.05 → significant.
       For discrete features, the KS-test reconstructs a baseline sample
       from the stored frequency table.

Additionally tracks:
  - Prediction confidence degradation
  - Default rate shift in the live prediction window
  - Per-feature Prometheus gauges for Grafana heatmap

Usage (standalone):
  python -m app.pipeline.drift_detector
  python -m app.pipeline.drift_detector --current-csv data/raw/credit_drifted.csv
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge
from scipy.stats import ks_2samp

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA_PATH         = Path(os.getenv("SCHEMA_PATH", "data/processed/feature_schema.json"))
DRIFT_WINDOW        = int(os.getenv("DRIFT_WINDOW",    "300"))
DRIFT_INTERVAL      = int(os.getenv("DRIFT_INTERVAL",  "120"))
PSI_WARN_THRESHOLD  = float(os.getenv("PSI_WARN_THRESHOLD",  "0.10"))
PSI_ALERT_THRESHOLD = float(os.getenv("PSI_ALERT_THRESHOLD", "0.25"))
KS_ALPHA            = float(os.getenv("KS_ALPHA", "0.05"))
MIN_WINDOW_SIZE     = int(os.getenv("MIN_WINDOW_SIZE", "50"))

FEATURE_COLUMNS = [
    "age", "income", "loan_amount", "credit_score",
    "debt_to_income", "employment_years", "num_credit_lines", "missed_payments",
]

# Integer-valued features — use frequency-table PSI, not percentile-bin PSI
DISCRETE_FEATURES = {"num_credit_lines", "missed_payments"}

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

DRIFT_PSI = Gauge(
    "ml_feature_drift_psi",
    "Population Stability Index per feature (vs training baseline)",
    ["feature"],
)
DRIFT_KS_STAT = Gauge(
    "ml_feature_drift_ks_statistic",
    "KS-test statistic per feature",
    ["feature"],
)
DRIFT_KS_PVALUE = Gauge(
    "ml_feature_drift_ks_pvalue",
    "KS-test p-value per feature",
    ["feature"],
)
DRIFT_DETECTED = Gauge(
    "ml_drift_detected",
    "1 if any feature exceeds PSI alert threshold, 0 otherwise",
)
DRIFT_FEATURES_COUNT = Gauge(
    "ml_drift_features_count",
    "Number of features with PSI above alert threshold",
)
PREDICTION_CONFIDENCE_MEAN = Gauge(
    "ml_prediction_confidence_mean",
    "Rolling mean prediction probability",
)
PREDICTION_CONFIDENCE_STD = Gauge(
    "ml_prediction_confidence_std",
    "Rolling std of prediction probabilities",
)
PREDICTION_DEFAULT_RATE = Gauge(
    "ml_prediction_default_rate",
    "Rolling default rate in recent prediction window",
)
DRIFT_CHECK_ERRORS  = Counter("ml_drift_check_errors_total",  "Drift check exceptions")
DRIFT_CHECKS_TOTAL  = Counter("ml_drift_checks_total",        "Total drift checks completed")
DRIFT_WINDOW_SIZE   = Gauge("ml_drift_window_size",           "Predictions in current drift window")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    feature:   str
    psi:       float
    ks_stat:   float
    ks_pvalue: float
    severity:  str    # "stable" | "warning" | "alert"


@dataclass
class DriftReport:
    timestamp:        float
    window_size:      int
    feature_results:  list[FeatureDriftResult] = field(default_factory=list)
    drifted_features: list[str]                = field(default_factory=list)
    overall_drift:    bool                     = False
    confidence_mean:  float                    = 0.0
    confidence_std:   float                    = 0.0
    default_rate:     float                    = 0.0
    error:            str | None               = None

    def summary(self) -> str:
        if self.error:
            return f"DriftReport[ERROR: {self.error}]"
        return (
            f"DriftReport[window={self.window_size}  "
            f"drift={self.overall_drift}  "
            f"features={self.drifted_features}  "
            f"conf={self.confidence_mean:.3f}]"
        )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def load_schema(path: Path = SCHEMA_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature schema not found: {path}\n"
            "Run: python -m app.pipeline.train"
        )
    schema = json.loads(path.read_text())
    log.info("Drift detector loaded schema from %s", path)
    return schema


# ---------------------------------------------------------------------------
# PSI — continuous features
# ---------------------------------------------------------------------------

def _psi_continuous(current_col: np.ndarray, psi_bins: list[float]) -> float:
    """
    PSI for continuous features using training-set percentile bins.

    Because psi_bins are percentile breakpoints of the training data,
    the baseline mass per bin is exactly 1/n_bins by construction.
    We only need to bin the current data.
    """
    edges = np.array(psi_bins, dtype=float)
    # Deduplicate while preserving order (handles near-duplicate percentiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    n_bins = len(edges) - 1
    inf_edges       = edges.copy()
    inf_edges[0]    = -np.inf
    inf_edges[-1]   =  np.inf

    c_counts = np.histogram(current_col, bins=inf_edges)[0]
    c_pct    = c_counts / len(current_col)
    b_pct    = np.full(n_bins, 1.0 / n_bins)

    eps   = 1e-6
    b_pct = np.clip(b_pct, eps, None)
    c_pct = np.clip(c_pct, eps, None)

    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


# ---------------------------------------------------------------------------
# PSI — discrete features
# ---------------------------------------------------------------------------

def _psi_discrete(current_col: np.ndarray, freq_table: dict[str, float]) -> float:
    """
    Frequency-table PSI for integer-valued features.

    freq_table maps str(value) → proportion in training data.
    Handles values not seen in training by grouping them into an
    'other' bucket (contributes to PSI but doesn't crash).
    """
    all_vals = sorted(set(int(v) for v in freq_table) |
                      set(int(v) for v in np.unique(current_col)))

    b_pct_list = []
    c_counts   = []

    for v in all_vals:
        b_pct_list.append(freq_table.get(str(v), 1e-6))
        c_counts.append(float(np.sum(current_col == v)))

    b_pct = np.array(b_pct_list, dtype=float)
    c_pct = np.array(c_counts,   dtype=float) / len(current_col)

    # Normalise baseline (should already sum to ~1, but guard for float drift)
    b_pct = b_pct / b_pct.sum()

    eps   = 1e-6
    b_pct = np.clip(b_pct, eps, None)
    c_pct = np.clip(c_pct, eps, None)

    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


# ---------------------------------------------------------------------------
# KS-test baseline reconstruction
# ---------------------------------------------------------------------------

def _ks_baseline_continuous(stats: dict, n: int = 1000) -> np.ndarray:
    """Reconstruct a continuous baseline sample by uniform-sampling within psi_bins."""
    psi_bins = stats["psi_bins"]
    edges    = np.array(psi_bins, dtype=float)
    edges    = np.unique(edges)
    n_bins   = len(edges) - 1
    if n_bins < 1:
        return np.linspace(stats["min"], stats["max"], n)

    per_bin = max(n // n_bins, 1)
    rng     = np.random.default_rng(0)
    samples = []
    for i in range(n_bins):
        lo = max(edges[i],   stats["min"])
        hi = min(edges[i+1], stats["max"])
        if lo >= hi:
            lo, hi = stats["min"], stats["max"]
        samples.append(rng.uniform(lo, hi, per_bin))
    return np.concatenate(samples)


def _ks_baseline_discrete(freq_table: dict[str, float], n: int = 1000) -> np.ndarray:
    """Reconstruct a discrete baseline sample from the training frequency table."""
    vals  = np.array([int(k) for k in freq_table.keys()])
    probs = np.array(list(freq_table.values()), dtype=float)
    probs = probs / probs.sum()
    rng   = np.random.default_rng(0)
    return rng.choice(vals, size=n, p=probs).astype(float)


# ---------------------------------------------------------------------------
# Core drift check
# ---------------------------------------------------------------------------

def check_drift(current_df: pd.DataFrame, schema: dict) -> DriftReport:
    n      = len(current_df)
    report = DriftReport(timestamp=time.time(), window_size=n)

    if n < MIN_WINDOW_SIZE:
        report.error = f"Window too small ({n} < {MIN_WINDOW_SIZE})"
        log.warning("Drift check skipped: %s", report.error)
        return report

    schema_features  = schema.get("features", {})
    drifted_features: list[str] = []

    for feature in FEATURE_COLUMNS:
        if feature not in schema_features or feature not in current_df.columns:
            log.warning("Skipping '%s': missing from schema or data", feature)
            continue

        stats       = schema_features[feature]
        current_col = current_df[feature].dropna().to_numpy()
        if len(current_col) < 10:
            continue

        # ── PSI ───────────────────────────────────────────────────────────
        if feature in DISCRETE_FEATURES:
            freq_table = stats.get("freq_table", {})
            if freq_table:
                psi = _psi_discrete(current_col, freq_table)
                baseline_ks = _ks_baseline_discrete(freq_table)
            else:
                # freq_table absent (old schema) — fall back to continuous PSI
                log.warning(
                    "freq_table missing for '%s' — re-run train.py to update schema",
                    feature,
                )
                psi = _psi_continuous(current_col, stats["psi_bins"])
                baseline_ks = _ks_baseline_continuous(stats)
        else:
            psi = _psi_continuous(current_col, stats["psi_bins"])
            baseline_ks = _ks_baseline_continuous(stats)

        # ── KS-test ───────────────────────────────────────────────────────
        ks_stat, ks_pvalue = ks_2samp(baseline_ks, current_col)

        # ── Severity ──────────────────────────────────────────────────────
        if psi >= PSI_ALERT_THRESHOLD:
            severity = "alert"
            drifted_features.append(feature)
        elif psi >= PSI_WARN_THRESHOLD or ks_pvalue < KS_ALPHA:
            severity = "warning"
        else:
            severity = "stable"

        report.feature_results.append(FeatureDriftResult(
            feature=feature, psi=psi,
            ks_stat=ks_stat, ks_pvalue=ks_pvalue,
            severity=severity,
        ))

        DRIFT_PSI.labels(feature=feature).set(psi)
        DRIFT_KS_STAT.labels(feature=feature).set(ks_stat)
        DRIFT_KS_PVALUE.labels(feature=feature).set(ks_pvalue)

    report.drifted_features = drifted_features
    report.overall_drift    = len(drifted_features) > 0

    DRIFT_DETECTED.set(1 if report.overall_drift else 0)
    DRIFT_FEATURES_COUNT.set(len(drifted_features))
    DRIFT_WINDOW_SIZE.set(n)

    return report


# ---------------------------------------------------------------------------
# Confidence metrics
# ---------------------------------------------------------------------------

def update_confidence_metrics(
    probabilities: list[float],
    decisions:     list[int],
) -> tuple[float, float, float]:
    if not probabilities:
        return 0.0, 0.0, 0.0
    arr  = np.array(probabilities)
    mean = float(arr.mean())
    std  = float(arr.std())
    dr   = float(np.mean([d == 1 for d in decisions])) if decisions else 0.0
    PREDICTION_CONFIDENCE_MEAN.set(mean)
    PREDICTION_CONFIDENCE_STD.set(std)
    PREDICTION_DEFAULT_RATE.set(dr)
    return mean, std, dr


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_report(report: DriftReport) -> None:
    if report.error:
        log.warning("Drift check: %s", report.error)
        return
    log.info("─" * 64)
    log.info("Drift check  window=%d  overall_drift=%s  ts=%.0f",
             report.window_size, report.overall_drift, report.timestamp)
    log.info("%-22s  %7s  %7s  %8s  %s",
             "Feature", "PSI", "KS-stat", "KS-pval", "Severity")
    log.info("─" * 64)
    for r in report.feature_results:
        marker = {"alert": "🔴", "warning": "🟡", "stable": "🟢"}[r.severity]
        log.info("%-22s  %7.4f  %7.4f  %8.4f  %s %s",
                 r.feature, r.psi, r.ks_stat, r.ks_pvalue, marker, r.severity)
    log.info("─" * 64)
    if report.drifted_features:
        log.warning("DRIFT ALERT: %s", ", ".join(report.drifted_features))
    log.info("Confidence  mean=%.4f  std=%.4f  default_rate=%.4f",
             report.confidence_mean, report.confidence_std, report.default_rate)


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

class DriftDetector:
    """Background drift loop — started by app/serving/app.py lifespan."""

    def __init__(
        self,
        schema:   dict | None = None,
        window:   int = DRIFT_WINDOW,
        interval: int = DRIFT_INTERVAL,
    ) -> None:
        self._schema   = schema or load_schema()
        self._window   = window
        self._interval = interval
        self._thread:  threading.Thread | None = None
        self._stop     = threading.Event()
        self._last_report: DriftReport | None  = None
        self._lock     = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="drift-detector", daemon=True,
        )
        self._thread.start()
        log.info("DriftDetector started  window=%d  interval=%ds",
                 self._window, self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("DriftDetector stopped")

    def get_last_report(self) -> DriftReport | None:
        with self._lock:
            return self._last_report

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_check()
            except Exception as exc:
                DRIFT_CHECK_ERRORS.inc()
                log.exception("Drift loop error: %s", exc)
            self._stop.wait(self._interval)

    def _run_check(self) -> None:
        from app.pipeline.predict import get_recent_predictions
        records = get_recent_predictions(self._window)
        if not records:
            log.debug("Drift check: prediction buffer empty")
            return
        rows = [r.features for r in records]
        df   = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        report = check_drift(df, self._schema)
        probs  = [r.probability for r in records]
        decs   = [r.decision    for r in records]
        mean, std, dr = update_confidence_metrics(probs, decs)
        report.confidence_mean = mean
        report.confidence_std  = std
        report.default_rate    = dr
        _log_report(report)
        DRIFT_CHECKS_TOTAL.inc()
        with self._lock:
            self._last_report = report


# ---------------------------------------------------------------------------
# Standalone batch report
# ---------------------------------------------------------------------------

def run_batch_report(
    current_csv: Path,
    schema_path: Path = SCHEMA_PATH,
    window:      int  = 0,
) -> DriftReport:
    schema = load_schema(schema_path)
    df     = pd.read_csv(current_csv)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    if window > 0 and len(df) > window:
        df = df.sample(n=window, random_state=42)
    df_feat = df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
    report = check_drift(df_feat, schema)
    if "default" in df.columns:
        dr = float(df["default"].mean())
        report.default_rate = dr
        PREDICTION_DEFAULT_RATE.set(dr)
    _log_report(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Standalone drift report against a CSV of inference records."
    )
    p.add_argument("--current-csv", type=Path, default=Path("data/raw/credit_drifted.csv"))
    p.add_argument("--schema-path", type=Path, default=SCHEMA_PATH)
    p.add_argument("--window",      type=int,  default=0,
                   help="Sample this many rows (0 = all)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args   = _parse_args()
    report = run_batch_report(args.current_csv, args.schema_path, args.window)

    print("\n" + "=" * 64)
    print("DRIFT REPORT SUMMARY")
    print("=" * 64)
    print(f"  Window size:      {report.window_size}")
    print(f"  Overall drift:    {'YES ⚠' if report.overall_drift else 'NO ✓'}")
    print(f"  Drifted features: {report.drifted_features or 'none'}")
    print(f"  Default rate:     {report.default_rate:.3f}")
    print("=" * 64)

    if report.overall_drift:
        print("\nDRIFT DETECTED — features exceeding PSI > 0.25:")
        for r in report.feature_results:
            if r.severity == "alert":
                print(f"  {r.feature:<22}  PSI={r.psi:.4f}  KS-p={r.ks_pvalue:.4f}")
        sys.exit(1)

if __name__ == "__main__":
    main()
