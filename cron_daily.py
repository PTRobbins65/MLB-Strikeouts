"""
MLB Strikeout Pipeline — Daily Cron Entrypoint

The single command a Railway *cron service* runs once per day, isolated from
the web dyno. It performs the two write-side jobs the read-only API depends on,
each as its OWN subprocess so the heavy phases never stack in one process:

  1. daily_snapshot.py — build the league Statcast snapshot (pitch-level "stuff"
     + batter discipline) and upload it to object storage. This process does NOT
     import xgboost/sklearn, so its memory baseline is small.
  2. pipeline.py       — run inference for today and write features_{date}.parquet
     (+ publish to storage) for the API to serve.

Why subprocesses
----------------
  * Memory isolation: a constrained container can't hold the ML stack AND a
    multi-month Statcast DataFrame in one process. Splitting them keeps each
    peak low.
  * Fault isolation: an out-of-memory SIGKILL during the snapshot build cannot
    be caught by try/except (the kernel kills the process). Running it as a
    separate process means such a kill is just a non-zero return code here — the
    inference step still runs, serving off the latest snapshot already in
    storage. The pipeline degrades gracefully on a stale/missing snapshot.

Exit codes
----------
  0  inference succeeded (features written for the API)
  1  inference failed
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from config import LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "cron_daily.log"),
    ],
)
logger = logging.getLogger("cron_daily")

ROOT = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description="Daily MLB strikeout cron job")
    p.add_argument("--date", type=str, help="Run date YYYY-MM-DD (default today)")
    p.add_argument("--history", type=int, default=0,
                   help="Seasons of PRIOR-year lookback for the SNAPSHOT Statcast pull "
                        "(default 0 = current season only). Kept small so the league pull "
                        "fits in a constrained container; rolling 'stuff' only needs recent starts.")
    p.add_argument("--infer-history", type=int, default=1,
                   help="Seasons of lookback for INFERENCE (StatsAPI season stats + game logs). "
                        "Default 1 = current + prior season, enough for season-stat fallback "
                        "and 20-start rolling windows.")
    p.add_argument("--skip-snapshot", action="store_true",
                   help="Skip the Statcast snapshot build (reuse the latest in storage)")
    return p.parse_args()


def _run_step(cmd, label: str) -> int:
    """Run one pipeline phase as an isolated subprocess (stdout streams to logs)."""
    logger.info(f"[{label}] starting: {' '.join(cmd)}")
    env = {**os.environ, "PYTHONUTF8": "1"}
    try:
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        logger.info(f"[{label}] exited with code {result.returncode}")
        return result.returncode
    except Exception as exc:
        logger.error(f"[{label}] failed to launch: {exc}")
        return 1


def main() -> int:
    args     = parse_args()
    target   = date.fromisoformat(args.date) if args.date else date.today()
    date_str = target.isoformat()
    logger.info(f"=== Daily cron run for {target} ===")

    # ── 1. Snapshot (isolated, best-effort) ────────────────────────────────
    # Even an OOM SIGKILL here is just a non-zero rc — inference still runs and
    # serves off the most recent snapshot already in storage.
    if args.skip_snapshot:
        logger.info("Snapshot build skipped (--skip-snapshot)")
    else:
        rc = _run_step(
            [sys.executable, "daily_snapshot.py",
             "--date", date_str, "--history", str(args.history)],
            "snapshot",
        )
        if rc != 0:
            logger.warning(
                f"Snapshot step failed (rc={rc}); inference will use the latest "
                f"snapshot already in storage"
            )

    # ── 2. Inference (isolated, must succeed) ──────────────────────────────
    rc = _run_step(
        [sys.executable, "pipeline.py",
         "--date", date_str, "--no-wait", "--history", str(args.infer_history)],
        "inference",
    )
    if rc != 0:
        logger.error(f"Inference step failed (rc={rc}) — nothing new for the API to serve")
        return 1

    logger.info(f"Daily cron complete for {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
