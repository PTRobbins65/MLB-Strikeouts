"""
MLB Strikeout Pipeline — Daily Statcast Snapshot Worker

The ONLY place Statcast pitch-level data is fetched. Runs once per day (Railway
cron, isolated from the web dyno), pulls one league-wide Statcast window, reduces
it to compact aggregates, and uploads them to object storage. The inference
pipeline then reads these aggregates instead of scraping Baseball Savant
per-pitcher on every request.

Artifacts written (via storage.py)
----------------------------------
  snapshots/pitcher_starts_{YYYYMMDD}.parquet
      One row per pitcher-start with the pitch-level "stuff" metrics
      (whiff_pct, csw_pct, zone_pct, o_swing_pct, avg_velo). FeatureBuilder
      rolls these into the whiff_pct_*, csw_pct_*, etc. windows.

  snapshots/batter_stuff_{YYYYMMDD}.parquet
      One row per (batter, season) with plate-discipline (o_swing_pct,
      contact_pct, swstr_pct) for opponent-lineup features.

Usage
-----
    python daily_snapshot.py                       # build today's snapshot
    python daily_snapshot.py --date 2024-11-01     # build as-of a past date
    python daily_snapshot.py --start 2024-03-01 --end 2024-11-01
    python daily_snapshot.py --history 2            # seasons of lookback
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

from config import LOG_DIR
from data_fetcher import HistoricalDataFetcher
from storage import get_storage

logger = logging.getLogger("daily_snapshot")

PITCHER_STARTS_PREFIX = "snapshots/pitcher_starts_"
BATTER_STUFF_PREFIX   = "snapshots/batter_stuff_"


class SnapshotBuilder:
    """Builds and uploads the daily Statcast aggregate snapshot."""

    def __init__(self, history_years: int = 2):
        self.history_years = history_years
        self.fetcher = HistoricalDataFetcher()
        self.storage = get_storage()

    def _resolve_window(
        self,
        target_date: date,
        start: Optional[str],
        end: Optional[str],
    ) -> Tuple[str, str]:
        """Return (start_dt, end_dt) strings for the Statcast pull."""
        if start and end:
            return start, end
        end_dt = target_date - timedelta(days=1)
        # Cover enough prior seasons that rolling windows have history early
        start_year = end_dt.year - self.history_years
        return f"{start_year}-03-01", end_dt.strftime("%Y-%m-%d")

    def build(
        self,
        target_date: Optional[date] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> bool:
        """Pull league Statcast, compute aggregates, upload snapshot. Returns ok."""
        target_date = target_date or date.today()
        start_dt, end_dt = self._resolve_window(target_date, start, end)
        logger.info(f"Building snapshot as-of {target_date} from Statcast {start_dt} -> {end_dt}")

        pitches = self.fetcher.get_statcast_date_range(start_dt, end_dt, force_refresh=force_refresh)
        if pitches is None or pitches.empty:
            logger.error("No league Statcast available — snapshot NOT written")
            return False
        logger.info(f"League Statcast loaded: {len(pitches):,} pitches")

        # Pitch-level per-start "stuff" metrics for every pitcher in the window
        starts = self.fetcher.compute_per_start_metrics(pitches)
        # Batter plate-discipline (o_swing/contact/swstr) per (batter, season)
        batter_stuff = self.fetcher.compute_batter_season_stats(pitches)

        if starts.empty or batter_stuff.empty:
            logger.error("Aggregation produced empty tables — snapshot NOT written")
            return False

        datetag = target_date.strftime("%Y%m%d")
        starts_key = f"{PITCHER_STARTS_PREFIX}{datetag}.parquet"
        batter_key = f"{BATTER_STUFF_PREFIX}{datetag}.parquet"

        self.storage.put_parquet(starts, starts_key)
        self.storage.put_parquet(batter_stuff, batter_key)

        logger.info(
            f"Snapshot {datetag} written: {len(starts):,} pitcher-starts, "
            f"{len(batter_stuff):,} batter-seasons"
        )
        return True


def load_latest_snapshot(storage=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (pitcher_starts, batter_stuff) from the most recent snapshot in
    storage. Used by the inference pipeline. Returns empty DataFrames if no
    snapshot exists yet (pipeline then degrades gracefully).
    """
    storage = storage or get_storage()
    starts_key = storage.latest(PITCHER_STARTS_PREFIX)
    batter_key = storage.latest(BATTER_STUFF_PREFIX)

    starts  = storage.get_parquet(starts_key)  if starts_key else None
    batters = storage.get_parquet(batter_key)  if batter_key else None

    if starts_key:
        logger.info(f"Loaded snapshot pitcher_starts: {starts_key} ({0 if starts is None else len(starts):,} rows)")
    else:
        logger.warning("No pitcher_starts snapshot found in storage")
    if batter_key:
        logger.info(f"Loaded snapshot batter_stuff: {batter_key} ({0 if batters is None else len(batters):,} rows)")
    else:
        logger.warning("No batter_stuff snapshot found in storage")

    return (
        starts if starts is not None else pd.DataFrame(),
        batters if batters is not None else pd.DataFrame(),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build the daily Statcast snapshot")
    p.add_argument("--date",    type=str, help="As-of date YYYY-MM-DD (default today)")
    p.add_argument("--start",   type=str, help="Explicit Statcast window start YYYY-MM-DD")
    p.add_argument("--end",     type=str, help="Explicit Statcast window end YYYY-MM-DD")
    p.add_argument("--history", type=int, default=2, help="Seasons of lookback (default 2)")
    p.add_argument("--refresh", action="store_true", help="Force re-fetch league Statcast")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "daily_snapshot.log"),
        ],
    )
    args = parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()

    builder = SnapshotBuilder(history_years=args.history)
    ok = builder.build(
        target_date=target,
        start=args.start,
        end=args.end,
        force_refresh=args.refresh,
    )
    sys.exit(0 if ok else 1)
