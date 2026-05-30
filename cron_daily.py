"""
MLB Strikeout Pipeline — Daily Cron Entrypoint

The single command a Railway *cron service* runs once per day, isolated from
the web dyno. It performs the two write-side jobs the read-only API depends on:

  1. Build the league-wide Statcast snapshot (daily_snapshot.SnapshotBuilder)
     → uploads pitcher-start "stuff" + batter plate-discipline to object storage.
  2. Run the inference pipeline for today (pipeline.DailyPipeline, no lineup wait)
     → writes features_{date}.parquet with predicted_k for the API to serve.

Design
------
  * This is the ONLY place league Statcast is fetched (one big pull/day), and
    the ONLY writer of the snapshot. The web dyno never scrapes — it reads the
    snapshot + features that this job produces.
  * The snapshot step failing (e.g. Baseball Savant hiccup) does NOT abort the
    run: the pipeline degrades gracefully on a stale/missing snapshot (pitch
    "stuff" features go NaN; StatsAPI counting features still drive predictions).
  * Intended to be triggered by Railway's cron schedule (see railway.toml). Also
    runnable by hand: `python cron_daily.py`.

Exit codes
----------
  0  features were written for today (predictions available)
  1  hard failure — no features produced
"""

import argparse
import logging
import sys
from datetime import date

from config import LOG_DIR
from daily_snapshot import SnapshotBuilder
from pipeline import DailyPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "cron_daily.log"),
    ],
)
logger = logging.getLogger("cron_daily")


def parse_args():
    p = argparse.ArgumentParser(description="Daily MLB strikeout cron job")
    p.add_argument("--date", type=str, help="Run date YYYY-MM-DD (default today)")
    p.add_argument("--history", type=int, default=2, help="Seasons of lookback (default 2)")
    p.add_argument("--skip-snapshot", action="store_true",
                   help="Skip the Statcast snapshot build (reuse the latest in storage)")
    return p.parse_args()


def main() -> int:
    args   = parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()
    logger.info(f"=== Daily cron run for {target} ===")

    # ── 1. Statcast snapshot (best-effort — never aborts the run) ───────────
    if args.skip_snapshot:
        logger.info("Snapshot build skipped (--skip-snapshot)")
    else:
        try:
            ok = SnapshotBuilder(history_years=args.history).build(target_date=target)
            if ok:
                logger.info("Snapshot build OK")
            else:
                logger.warning("Snapshot build returned no data — pipeline will use the latest available snapshot")
        except Exception as exc:
            logger.warning(f"Snapshot build failed ({exc}); continuing with the latest snapshot in storage")

    # ── 2. Inference pipeline (the job that actually must succeed) ──────────
    try:
        pipeline = DailyPipeline(target_date=target, history_years=args.history)
        features = pipeline.run(wait_for_lineups=False)
    except Exception as exc:
        logger.error(f"Pipeline run failed: {exc}", exc_info=True)
        return 1

    if features is None or features.empty:
        logger.error("No features produced — nothing for the API to serve")
        return 1

    has_preds = "predicted_k" in features.columns
    logger.info(
        f"Daily cron complete for {target}: {len(features)} pitcher rows"
        + (" with predictions" if has_preds else " (no model — predictions skipped)")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
