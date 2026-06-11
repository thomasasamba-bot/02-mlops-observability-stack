"""
scripts/data/generate_data.py

Synthetic credit scoring dataset generator.
Produces two datasets:
  - baseline: stable feature distributions used for model training
  - drifted:  shifted distributions used for drift detection testing

Features (8 total):
  age                 — applicant age (years)
  income              — annual income (USD)
  loan_amount         — requested loan (USD)
  credit_score        — FICO-style score 300-850
  debt_to_income      — ratio 0.0-1.0
  employment_years    — years at current employer
  num_credit_lines    — open credit lines
  missed_payments     — missed payments in last 24 months

Target:
  default             — 1 = default, 0 = no default (~20% positive rate)

Label generation strategy:
  Labels are assigned DETERMINISTICALLY from a weighted risk score
  (threshold at 80th percentile → 20% default rate), then 8% label
  noise is added to keep the dataset realistic and non-trivially separable.
  This yields a RandomForest ROC-AUC of ~0.84 with well-spread probabilities,
  making model drift and confidence degradation clearly observable.

Drift injected (drifted dataset):
  - income:          mean drops ~25% (economic shock)
  - credit_score:    mean drops ~40 points (portfolio degradation)
  - debt_to_income:  mean rises ~0.14 (higher leverage)
  - missed_payments: rate increases ~2.75× (lagging indicator)
  These shifts produce PSI > 0.2 and KS-test p < 0.05 on all four features.

Usage:
  python scripts/data/generate_data.py
  python scripts/data/generate_data.py --output-dir data/raw --samples 10000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

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

# Label noise fraction — makes dataset realistic (not perfectly separable)
LABEL_NOISE_RATE = 0.08


# ---------------------------------------------------------------------------
# Feature sampling
# ---------------------------------------------------------------------------

def _sample_features(rng: np.random.Generator, n: int, drift: bool = False) -> pd.DataFrame:
    """Draw n feature rows. If drift=True, apply distribution shifts."""

    # ── Credit score ──────────────────────────────────────────────────────
    cs_mean = 638.0 if drift else 680.0
    cs_std  = 85.0  if drift else 75.0
    credit_score = np.clip(rng.normal(cs_mean, cs_std, n), 300, 850)

    # ── Income ────────────────────────────────────────────────────────────
    # Drift: lognormal mean shifts from ln(54k)≈10.9 to ln(40k)≈10.6
    inc_mean = 10.60 if drift else 10.90
    inc_std  = 0.75  if drift else 0.60
    income = np.clip(rng.lognormal(inc_mean, inc_std, n), 15_000, 500_000)

    # ── Loan amount ───────────────────────────────────────────────────────
    loan_amount = np.clip(rng.lognormal(9.9, 0.7, n), 1_000, 100_000)

    # ── Debt-to-income ────────────────────────────────────────────────────
    # Drift: Beta(2.5,5.0) → Beta(3.5,4.0) shifts distribution right
    dti_a = 3.5 if drift else 2.5
    dti_b = 4.0 if drift else 5.0
    debt_to_income = np.clip(rng.beta(dti_a, dti_b, n), 0.01, 0.95)

    # ── Employment years ──────────────────────────────────────────────────
    employment_years = np.clip(rng.gamma(2.5, 3.0, n), 0, 40)

    # ── Age ───────────────────────────────────────────────────────────────
    age = np.clip(rng.normal(38.0, 10.0, n), 18, 80)

    # ── Credit lines ──────────────────────────────────────────────────────
    num_credit_lines = np.clip(rng.poisson(4.5, n), 0, 20).astype(float)

    # ── Missed payments ───────────────────────────────────────────────────
    # Drift: Poisson(0.4) → Poisson(1.1)
    mp_lam = 1.1 if drift else 0.4
    missed_payments = np.clip(rng.poisson(mp_lam, n), 0, 10).astype(float)

    return pd.DataFrame({
        "age":              age,
        "income":           income,
        "loan_amount":      loan_amount,
        "credit_score":     credit_score,
        "debt_to_income":   debt_to_income,
        "employment_years": employment_years,
        "num_credit_lines": num_credit_lines,
        "missed_payments":  missed_payments,
    })


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _assign_labels(df: pd.DataFrame, rng: np.random.Generator, noise_rate: float) -> np.ndarray:
    """
    Deterministic risk score → threshold at 80th percentile → 20% default rate.
    Then flip `noise_rate` fraction of labels to add realistic ambiguity.

    This strategy gives the RandomForest a clean signal to learn from,
    yielding ROC-AUC ~0.84 and well-spread probability outputs.
    """
    n = len(df)

    # Normalise each driver to [0, 1]
    cs_norm  = (df["credit_score"] - 300) / 550          # 0=worst credit, 1=best
    dti_norm = df["debt_to_income"]                       # already [0,1]
    mp_norm  = np.minimum(df["missed_payments"] / 4.0, 1.0)
    lti      = np.minimum(df["loan_amount"] / df["income"], 1.5) / 1.5
    inc_norm = 1.0 - np.minimum(df["income"] / 120_000, 1.0)
    ten_norm = 1.0 - np.minimum(df["employment_years"] / 10.0, 1.0)

    # Weighted risk score — higher = riskier
    score = (
        - 3.5 * cs_norm       # credit score: strongest signal
        + 2.0 * mp_norm       # missed payments: near-deterministic
        + 1.8 * dti_norm      # debt burden
        + 1.2 * lti           # loan-to-income ratio
        + 0.8 * inc_norm      # income level
        + 0.5 * ten_norm      # employment stability
    ).to_numpy()

    # Deterministic labels at the 80th percentile → 20% defaults
    threshold = np.percentile(score, 80)
    labels = (score >= threshold).astype(int)

    # Add label noise
    noise_mask = rng.uniform(size=n) < noise_rate
    labels[noise_mask] = 1 - labels[noise_mask]

    return labels


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def generate_baseline(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = _sample_features(rng, n, drift=False)
    df[TARGET_COLUMN] = _assign_labels(df, rng, LABEL_NOISE_RATE)

    log.info(
        "Baseline  — n=%d  default_rate=%.1f%%  "
        "income_median=%.0f  credit_score_mean=%.1f  dti_mean=%.3f",
        n, df[TARGET_COLUMN].mean() * 100,
        df["income"].median(), df["credit_score"].mean(), df["debt_to_income"].mean(),
    )
    return df


def generate_drifted(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = _sample_features(rng, n, drift=True)
    df[TARGET_COLUMN] = _assign_labels(df, rng, LABEL_NOISE_RATE)

    log.info(
        "Drifted   — n=%d  default_rate=%.1f%%  "
        "income_median=%.0f  credit_score_mean=%.1f  dti_mean=%.3f",
        n, df[TARGET_COLUMN].mean() * 100,
        df["income"].median(), df["credit_score"].mean(), df["debt_to_income"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Drift validation
# ---------------------------------------------------------------------------

def validate_drift(baseline: pd.DataFrame, drifted: pd.DataFrame) -> None:
    log.info("─" * 60)
    log.info("Drift validation (KS-test, α=0.05)")
    log.info("%-20s  %8s  %8s  %s", "Feature", "KS-stat", "p-value", "Drifted?")
    log.info("─" * 60)

    drifted_features = []
    for col in FEATURE_COLUMNS:
        stat, p = ks_2samp(baseline[col].to_numpy(), drifted[col].to_numpy())
        flag = "✓ DRIFT" if p < 0.05 else "  stable"
        if p < 0.05:
            drifted_features.append(col)
        log.info("%-20s  %8.4f  %8.4f  %s", col, stat, p, flag)

    log.info("─" * 60)
    log.info(
        "Summary: %d/%d features show significant drift: %s",
        len(drifted_features), len(FEATURE_COLUMNS),
        ", ".join(drifted_features) if drifted_features else "none",
    )


def compute_psi(baseline_col: np.ndarray, drifted_col: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.percentile(baseline_col, np.linspace(0, 100, bins + 1))
    breakpoints[0]  -= 1e-6
    breakpoints[-1] += 1e-6

    b_pct = np.histogram(baseline_col, bins=breakpoints)[0] / len(baseline_col)
    d_pct = np.histogram(drifted_col,  bins=breakpoints)[0] / len(drifted_col)

    eps   = 1e-6
    b_pct = np.clip(b_pct, eps, None)
    d_pct = np.clip(d_pct, eps, None)

    return float(np.sum((d_pct - b_pct) * np.log(d_pct / b_pct)))


def log_psi_summary(baseline: pd.DataFrame, drifted: pd.DataFrame) -> None:
    log.info("─" * 60)
    log.info("PSI summary (bins=10)")
    log.info("%-20s  %8s  %s", "Feature", "PSI", "Severity")
    log.info("─" * 60)
    for col in FEATURE_COLUMNS:
        psi = compute_psi(baseline[col].to_numpy(), drifted[col].to_numpy())
        if psi < 0.10:
            severity = "  stable"
        elif psi < 0.25:
            severity = "⚠ moderate"
        else:
            severity = "✗ SIGNIFICANT"
        log.info("%-20s  %8.4f  %s", col, psi, severity)
    log.info("─" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic credit-scoring datasets for ML pipeline."
    )
    parser.add_argument("--output-dir",    type=Path,  default=Path("data/raw"))
    parser.add_argument("--samples",       type=int,   default=5_000)
    parser.add_argument("--drift-samples", type=int,   default=1_000)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--validate",      action="store_true", default=True)
    parser.add_argument("--no-validate",   dest="validate", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    log.info("Generating datasets  seed=%d", args.seed)
    baseline = generate_baseline(args.samples,       rng)
    drifted  = generate_drifted(args.drift_samples,  rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "credit_baseline.csv"
    drifted_path  = args.output_dir / "credit_drifted.csv"

    baseline.to_csv(baseline_path, index=False)
    drifted.to_csv(drifted_path,   index=False)

    log.info("Written: %s  (%d rows)", baseline_path, len(baseline))
    log.info("Written: %s  (%d rows)", drifted_path,  len(drifted))

    if args.validate:
        validate_drift(baseline, drifted)
        log_psi_summary(baseline, drifted)

    log.info("Done.")


if __name__ == "__main__":
    main()