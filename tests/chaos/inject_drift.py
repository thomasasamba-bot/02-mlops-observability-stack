"""
tests/chaos/inject_drift.py

Chaos test: inject drifted inference traffic into the live serving endpoint.

What this does:
  1. Loads credit_drifted.csv (produced by generate_data.py)
  2. Sends records to POST /predict on the inference server in batches
  3. Waits for the DriftDetector background thread to complete a check cycle
  4. Polls GET /drift/status until overall_drift=True or timeout
  5. Exits 0 if drift is detected within the timeout, 1 otherwise

Purpose:
  - Validates the end-to-end drift detection pipeline under real traffic
  - Used in CI to confirm the detector fires when it should
  - Can be run manually to populate the prediction buffer for Grafana demos

Usage:
  # Requires inference server running on port 8006 and MLflow on port 5000
  python tests/chaos/inject_drift.py
  python tests/chaos/inject_drift.py --records 500 --batch-size 25 --timeout 300
  python tests/chaos/inject_drift.py --baseline   # inject stable traffic (no drift expected)
  python tests/chaos/inject_drift.py --dry-run    # validate CSV without sending requests
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_INFERENCE_URL = "http://localhost:8006"
DEFAULT_DRIFTED_CSV   = Path("data/raw/credit_drifted.csv")
DEFAULT_BASELINE_CSV  = Path("data/raw/credit_baseline.csv")
DEFAULT_RECORDS       = 300     # how many rows to inject
DEFAULT_BATCH_SIZE    = 20      # records per /predict/batch call
DEFAULT_TIMEOUT       = 300     # seconds to wait for drift detection
DEFAULT_POLL_INTERVAL = 10      # seconds between /drift/status polls
DEFAULT_DELAY         = 0.05    # seconds between batches (avoid overwhelming server)

FEATURE_COLUMNS = [
    "age", "income", "loan_amount", "credit_score",
    "debt_to_income", "employment_years", "num_credit_lines", "missed_payments",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: int = 10) -> dict:
    data    = json.dumps(payload).encode("utf-8")
    req     = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_server(base_url: str) -> bool:
    try:
        resp = _get(f"{base_url}/health/live")
        return resp.get("status") == "alive"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_records(csv_path: Path, n: int) -> list[dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Run: python scripts/data/generate_data.py"
        )

    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            for col in FEATURE_COLUMNS:
                if col not in row:
                    raise ValueError(f"CSV missing column: {col}")
                record[col] = float(row[col])
            records.append(record)
            if len(records) >= n:
                break

    if len(records) < n:
        log.warning(
            "Requested %d records but CSV only has %d rows — using all",
            n, len(records),
        )

    log.info("Loaded %d records from %s", len(records), csv_path)
    return records


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_records(
    records:    list[dict],
    base_url:   str,
    batch_size: int,
    delay:      float,
) -> dict[str, int]:
    """
    Send all records to the inference server in batches.
    Returns summary stats.
    """
    total        = len(records)
    sent         = 0
    defaults     = 0
    no_defaults  = 0
    low_conf     = 0
    errors       = 0
    batch_num    = 0

    batches = [records[i:i+batch_size] for i in range(0, total, batch_size)]
    log.info(
        "Injecting %d records in %d batches of %d …",
        total, len(batches), batch_size,
    )

    for batch in batches:
        batch_num += 1
        try:
            resp = _post(
                f"{base_url}/predict/batch",
                {"records": batch},
                timeout=15,
            )
            sent        += resp.get("total", 0)
            defaults    += resp.get("defaults", 0)
            no_defaults += resp.get("no_defaults", 0)
            low_conf    += resp.get("low_confidence", 0)

            if batch_num % 5 == 0 or batch_num == len(batches):
                log.info(
                    "  Batch %d/%d  sent=%d  defaults=%d  low_conf=%d",
                    batch_num, len(batches), sent, defaults, low_conf,
                )

        except Exception as exc:
            errors += 1
            log.warning("Batch %d failed: %s", batch_num, exc)

        if delay > 0:
            time.sleep(delay)

    log.info(
        "Injection complete  sent=%d  defaults=%d (%.1f%%)  "
        "no_defaults=%d  low_conf=%d  errors=%d",
        sent,
        defaults, (defaults / sent * 100) if sent else 0,
        no_defaults, low_conf, errors,
    )

    return {
        "sent":        sent,
        "defaults":    defaults,
        "no_defaults": no_defaults,
        "low_conf":    low_conf,
        "errors":      errors,
    }


# ---------------------------------------------------------------------------
# Drift polling
# ---------------------------------------------------------------------------

def wait_for_drift(
    base_url:      str,
    timeout:       int,
    poll_interval: int,
    expect_drift:  bool = True,
) -> bool:
    """
    Poll /drift/status until overall_drift matches expect_drift or timeout.

    Returns True if the expected outcome was observed, False on timeout.
    """
    deadline = time.time() + timeout
    attempt  = 0

    log.info(
        "Waiting up to %ds for drift_detected=%s  (poll every %ds) …",
        timeout, expect_drift, poll_interval,
    )

    while time.time() < deadline:
        attempt += 1
        try:
            status = _get(f"{base_url}/drift/status")

            detector = status.get("drift_detector", "unknown")
            drift    = status.get("overall_drift", False)
            features = status.get("drifted_features", [])
            conf     = status.get("confidence_mean", 0)

            log.info(
                "  Poll %d  detector=%s  drift=%s  features=%s  conf_mean=%.3f",
                attempt, detector, drift, features, conf,
            )

            if drift == expect_drift:
                log.info(
                    "✓ Expected drift=%s confirmed after %d poll(s)",
                    expect_drift, attempt,
                )
                return True

            # No check completed yet — wait a bit
            if detector == "running" and status.get("message"):
                log.info("  Drift check not yet completed, waiting …")

        except Exception as exc:
            log.warning("  Poll %d failed: %s", attempt, exc)

        time.sleep(poll_interval)

    log.error(
        "✗ Timeout after %ds waiting for drift=%s  (last drift=%s)",
        timeout, expect_drift, drift if "drift" in dir() else "unknown",
    )
    return False


# ---------------------------------------------------------------------------
# Full end-to-end run
# ---------------------------------------------------------------------------

def run(
    csv_path:      Path,
    base_url:      str,
    n_records:     int,
    batch_size:    int,
    timeout:       int,
    poll_interval: int,
    delay:         float,
    expect_drift:  bool,
    dry_run:       bool,
) -> int:
    """
    Run the full inject → wait → verify cycle.
    Returns exit code: 0 = success, 1 = failure.
    """
    # ── Pre-flight checks ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Drift Injection Chaos Test")
    log.info("  CSV:          %s", csv_path)
    log.info("  Records:      %d", n_records)
    log.info("  Batch size:   %d", batch_size)
    log.info("  Inference:    %s", base_url)
    log.info("  Expect drift: %s", expect_drift)
    log.info("  Dry run:      %s", dry_run)
    log.info("=" * 60)

    records = load_records(csv_path, n_records)

    if dry_run:
        log.info("Dry run — CSV validated, %d records ready. Exiting.", len(records))
        return 0

    if not _check_server(base_url):
        log.error(
            "Inference server not reachable at %s\n"
            "Start it with: uvicorn app.serving.app:app --port 8006 --workers 1",
            base_url,
        )
        return 1

    log.info("Inference server is up ✓")

    # ── Inject ───────────────────────────────────────────────────────────
    stats = inject_records(records, base_url, batch_size, delay)

    if stats["errors"] > len(records) * 0.1:
        log.error(
            "Too many injection errors (%d / %d) — aborting",
            stats["errors"], len(records),
        )
        return 1

    # ── Wait for drift detector cycle ────────────────────────────────────
    detected = wait_for_drift(base_url, timeout, poll_interval, expect_drift)

    # ── Final status ─────────────────────────────────────────────────────
    try:
        final = _get(f"{base_url}/drift/status")
        log.info("Final drift status:")
        log.info("  overall_drift:    %s", final.get("overall_drift"))
        log.info("  drifted_features: %s", final.get("drifted_features"))
        log.info("  confidence_mean:  %s", final.get("confidence_mean"))
        log.info("  default_rate:     %s", final.get("default_rate"))
    except Exception:
        pass

    if detected:
        log.info("✓ Chaos test PASSED — drift=%s as expected", expect_drift)
        return 0
    else:
        log.error("✗ Chaos test FAILED — drift=%s not observed within %ds",
                  expect_drift, timeout)
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chaos test: inject drifted traffic into the inference server "
            "and verify the drift detector fires."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DRIFTED_CSV,
        help=f"CSV to inject (default: {DEFAULT_DRIFTED_CSV})",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Inject baseline (stable) traffic instead — expect NO drift",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=DEFAULT_RECORDS,
        help=f"Number of records to inject (default: {DEFAULT_RECORDS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Records per batch request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--inference-url",
        type=str,
        default=DEFAULT_INFERENCE_URL,
        help=f"Inference server base URL (default: {DEFAULT_INFERENCE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Seconds to wait for drift detection (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between status polls (default: {DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between batch requests (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV only — do not send requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.baseline:
        csv_path     = DEFAULT_BASELINE_CSV
        expect_drift = False
    else:
        csv_path     = args.csv
        expect_drift = True

    exit_code = run(
        csv_path=csv_path,
        base_url=args.inference_url,
        n_records=args.records,
        batch_size=args.batch_size,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        delay=args.delay,
        expect_drift=expect_drift,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()