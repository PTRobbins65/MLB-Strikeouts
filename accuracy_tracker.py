"""
MLB Strikeout Pipeline — Accuracy Tracker

Logs daily predictions and fills in actual strikeout totals once games
complete.  Provides rolling MAE / RMSE / bias metrics so you can see
when the model is drifting and retraining is warranted.

Storage
-------
  data/accuracy_log.parquet  — one row per pitcher-start prediction

Schema
------
  game_date        : date of the game
  game_pk          : MLB game ID
  pitcher_mlbam    : pitcher MLBAM ID
  pitcher_name     : display name
  team             : pitcher's team
  opponent         : opposing team
  is_home          : True if pitcher's team is home
  predicted_k      : model point-estimate
  actual_k         : actual strikeouts (null until recorded)
  error            : predicted_k - actual_k (null until recorded)
  abs_error        : |error|
  lineup_confirmed : 1 if lineup was confirmed at prediction time
  model_version    : YYYYMMDD stamp of model file used
  recorded_at      : UTC timestamp when prediction was logged
  actuals_at       : UTC timestamp when actual was filled in

Usage
-----
  # Log today's predictions (call after pipeline.run())
  from accuracy_tracker import AccuracyTracker
  tracker = AccuracyTracker()
  tracker.log_predictions(features_df, model_version="20260522")

  # Record actuals for yesterday (call nightly)
  tracker.record_actuals("2026-05-21")

  # Get performance summary
  summary = tracker.get_summary(window_days=30)
  print(summary)

  # Full history as DataFrame
  log_df = tracker.get_log()
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from config import DATA_DIR
from data_fetcher import HistoricalDataFetcher

logger = logging.getLogger(__name__)

LOG_PATH = DATA_DIR / "accuracy_log.parquet"

# Columns stored in the log
LOG_SCHEMA = {
    "game_date":        "object",     # str "YYYY-MM-DD"
    "game_pk":          "int64",
    "pitcher_mlbam":    "int64",
    "pitcher_name":     "object",
    "team":             "object",
    "opponent":         "object",
    "is_home":          "bool",
    "predicted_k":      "float64",
    "actual_k":         "float64",    # NaN until recorded
    "error":            "float64",    # NaN until recorded
    "abs_error":        "float64",    # NaN until recorded
    "lineup_confirmed": "int64",
    "model_version":    "object",
    "recorded_at":      "object",     # ISO UTC string
    "actuals_at":       "object",     # ISO UTC string, NaN until recorded
}


class AccuracyTracker:
    """
    Persists predictions and actuals, computes rolling performance metrics.
    Thread-safe for reading; writes are serialised through pandas append/overwrite.
    """

    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self._fetcher = HistoricalDataFetcher()

    # ── Write: log predictions ─────────────────────────────────────────────

    def log_predictions(
        self,
        features_df: pd.DataFrame,
        model_version: str = "",
    ) -> int:
        """
        Append prediction rows for one game date.

        Parameters
        ----------
        features_df  : output from pipeline.run() — must contain predicted_k
        model_version: short string identifying which model was used
                       (e.g. "20260522").  Falls back to today's date if empty.

        Returns the number of rows appended.
        """
        if features_df.empty or "predicted_k" not in features_df.columns:
            logger.warning("log_predictions: nothing to log (no predicted_k column)")
            return 0

        version = model_version or date.today().strftime("%Y%m%d")
        now_str = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        rows = []
        for _, r in features_df.iterrows():
            game_date = str(r.get("game_date", ""))[:10]
            if not game_date:
                continue

            rows.append({
                "game_date":        game_date,
                "game_pk":          int(r.get("game_pk", 0) or 0),
                "pitcher_mlbam":    int(r.get("pitcher_mlbam", 0) or 0),
                "pitcher_name":     str(r.get("pitcher_name", "")),
                "team":             str(r.get("home_team", "") if r.get("is_home") else r.get("away_team", "")),
                "opponent":         str(r.get("away_team", "") if r.get("is_home") else r.get("home_team", "")),
                "is_home":          bool(r.get("is_home", True)),
                "predicted_k":      float(r["predicted_k"]),
                "actual_k":         np.nan,
                "error":            np.nan,
                "abs_error":        np.nan,
                "lineup_confirmed": int(r.get("lineup_confirmed", 0) or 0),
                "model_version":    version,
                "recorded_at":      now_str,
                "actuals_at":       "",
            })

        if not rows:
            return 0

        new_df = pd.DataFrame(rows)
        existing = self._load()

        # De-duplicate: if we already have a row for this game_pk + pitcher,
        # overwrite it (re-running pipeline before game start updates the lineup).
        if not existing.empty and "game_pk" in existing.columns:
            key = ["game_pk", "pitcher_mlbam"]
            existing = existing[
                ~existing.set_index(key).index.isin(new_df.set_index(key).index)
            ]

        combined = pd.concat([existing, new_df], ignore_index=True)
        self._save(combined)
        logger.info(f"Accuracy log: appended {len(new_df)} prediction rows -> {self.log_path.name}")
        return len(new_df)

    # ── Write: fill in actuals ─────────────────────────────────────────────

    def record_actuals(
        self,
        target_date: Union[str, date],
        force: bool = False,
    ) -> int:
        """
        Fetch Statcast data for target_date and fill actual_k for all
        predictions that don't yet have it.

        Parameters
        ----------
        target_date : "YYYY-MM-DD" or date object
        force       : if True, overwrite existing actuals (useful for corrections)

        Returns the number of rows updated.
        """
        date_str = str(target_date)[:10]
        existing = self._load()

        if existing.empty:
            logger.warning("record_actuals: accuracy log is empty — log predictions first")
            return 0

        # Rows for this date that still need actuals
        mask = existing["game_date"] == date_str
        if not force:
            mask = mask & existing["actual_k"].isna()

        pending = existing[mask]
        if pending.empty:
            logger.info(f"record_actuals: no pending rows for {date_str}")
            return 0

        # Fetch Statcast for the date
        logger.info(f"Fetching Statcast actuals for {date_str}...")
        try:
            pitches = self._fetcher.get_statcast_date_range(date_str, date_str)
        except Exception as exc:
            logger.error(f"record_actuals: Statcast fetch failed — {exc}")
            return 0

        if pitches.empty:
            logger.warning(f"record_actuals: no Statcast data for {date_str}")
            return 0

        # Aggregate actual Ks per pitcher per game
        actuals = self._compute_actuals(pitches)
        if actuals.empty:
            return 0

        # Join actuals back into the log
        now_str = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        updated = 0

        for idx in pending.index:
            row = existing.loc[idx]
            game_pk  = int(row["game_pk"])
            mlbam    = int(row["pitcher_mlbam"])

            match = actuals[
                (actuals["game_pk"] == game_pk) &
                (actuals["pitcher_mlbam"] == mlbam)
            ]
            if match.empty:
                # Try by pitcher only (game_pk=0 for legacy rows)
                match = actuals[actuals["pitcher_mlbam"] == mlbam]

            if match.empty:
                continue

            actual_k = float(match.iloc[0]["actual_k"])
            pred_k   = float(row["predicted_k"])
            existing.loc[idx, "actual_k"]   = actual_k
            existing.loc[idx, "error"]      = round(pred_k - actual_k, 2)
            existing.loc[idx, "abs_error"]  = round(abs(pred_k - actual_k), 2)
            existing.loc[idx, "actuals_at"] = now_str
            updated += 1

        if updated:
            self._save(existing)
            logger.info(
                f"record_actuals: filled {updated}/{len(pending)} rows for {date_str}"
            )
        else:
            logger.warning(
                f"record_actuals: no game_pk/pitcher matches found for {date_str} "
                "— game may still be in progress or pitcher didn't start"
            )

        return updated

    # ── Read: performance summary ──────────────────────────────────────────

    def get_summary(self, window_days: Optional[int] = None) -> dict:
        """
        Return a dict of performance metrics.

        Parameters
        ----------
        window_days : if set, restrict to the most recent N days.
                      None = all-time.

        Returns
        -------
        {
          "n_predictions" : int,   total rows in window
          "n_graded"      : int,   rows with actuals recorded
          "mae"           : float, mean absolute error
          "rmse"          : float, root mean squared error
          "bias"          : float, mean(predicted - actual) — positive = over-predicting
          "within_1_k"    : float, fraction where |error| <= 1.0
          "within_2_k"    : float, fraction where |error| <= 2.0
          "mae_confirmed" : float, MAE for confirmed-lineup predictions only
          "mae_projected" : float, MAE for projected-lineup predictions only
          "window_days"   : int | None,
          "start_date"    : str,
          "end_date"      : str,
        }
        """
        log = self._load()
        if log.empty:
            return self._empty_summary(window_days)

        if window_days is not None:
            cutoff = (date.today() - timedelta(days=window_days)).isoformat()
            log = log[log["game_date"] >= cutoff]

        graded = log.dropna(subset=["actual_k", "error"])

        result = {
            "n_predictions": len(log),
            "n_graded":      len(graded),
            "window_days":   window_days,
            "start_date":    log["game_date"].min() if len(log) else "",
            "end_date":      log["game_date"].max() if len(log) else "",
        }

        if graded.empty:
            result.update({"mae": None, "rmse": None, "bias": None,
                           "within_1_k": None, "within_2_k": None,
                           "mae_confirmed": None, "mae_projected": None})
            return result

        errors     = graded["error"].astype(float)
        abs_errors = graded["abs_error"].astype(float)

        result["mae"]       = round(float(abs_errors.mean()), 3)
        result["rmse"]      = round(float(np.sqrt((errors ** 2).mean())), 3)
        result["bias"]      = round(float(errors.mean()), 3)
        result["within_1_k"] = round(float((abs_errors <= 1.0).mean()), 3)
        result["within_2_k"] = round(float((abs_errors <= 2.0).mean()), 3)

        confirmed = graded[graded["lineup_confirmed"] == 1]["abs_error"].astype(float)
        projected = graded[graded["lineup_confirmed"] == 0]["abs_error"].astype(float)
        result["mae_confirmed"] = round(float(confirmed.mean()), 3) if len(confirmed) else None
        result["mae_projected"] = round(float(projected.mean()), 3) if len(projected) else None

        return result

    def get_daily_mae(self, window_days: Optional[int] = 90) -> pd.DataFrame:
        """
        Return a DataFrame with one row per game_date showing daily MAE and
        a 7-day rolling MAE — useful for charting trend in the UI.

        Columns: game_date, n_graded, daily_mae, rolling_mae_7d, bias
        """
        log = self._load()
        if log.empty:
            return pd.DataFrame(columns=["game_date", "n_graded", "daily_mae",
                                         "rolling_mae_7d", "bias"])

        graded = log.dropna(subset=["actual_k", "abs_error"]).copy()
        if graded.empty:
            return pd.DataFrame()

        if window_days is not None:
            cutoff = (date.today() - timedelta(days=window_days)).isoformat()
            graded = graded[graded["game_date"] >= cutoff]

        graded["abs_error"] = graded["abs_error"].astype(float)
        graded["error"]     = graded["error"].astype(float)

        daily = (
            graded.groupby("game_date")
            .agg(
                n_graded  = ("abs_error", "count"),
                daily_mae = ("abs_error", "mean"),
                bias      = ("error",     "mean"),
            )
            .reset_index()
            .sort_values("game_date")
        )

        daily["rolling_mae_7d"] = (
            daily["daily_mae"]
            .rolling(window=7, min_periods=1)
            .mean()
            .round(3)
        )
        daily["daily_mae"] = daily["daily_mae"].round(3)
        daily["bias"]      = daily["bias"].round(3)

        return daily

    def get_log(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        graded_only: bool = False,
    ) -> pd.DataFrame:
        """Return the raw accuracy log, optionally filtered by date range."""
        log = self._load()
        if log.empty:
            return log
        if start_date:
            log = log[log["game_date"] >= start_date]
        if end_date:
            log = log[log["game_date"] <= end_date]
        if graded_only:
            log = log.dropna(subset=["actual_k"])
        return log.sort_values("game_date").reset_index(drop=True)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame(columns=list(LOG_SCHEMA.keys()))
        try:
            return pd.read_parquet(self.log_path)
        except Exception as exc:
            logger.error(f"Could not read accuracy log: {exc}")
            return pd.DataFrame(columns=list(LOG_SCHEMA.keys()))

    def _save(self, df: pd.DataFrame):
        df.to_parquet(self.log_path, index=False)

    @staticmethod
    def _compute_actuals(pitches: pd.DataFrame) -> pd.DataFrame:
        """
        From raw Statcast pitch data, compute actual strikeout totals
        per pitcher per game.  Returns DataFrame with columns:
        game_pk, pitcher_mlbam, actual_k.
        """
        if pitches.empty:
            return pd.DataFrame()

        # A strikeout is recorded when events == 'strikeout'
        if "events" not in pitches.columns:
            return pd.DataFrame()

        ks = pitches[pitches["events"] == "strikeout"].copy()

        group_cols = []
        if "game_pk" in ks.columns:
            group_cols.append("game_pk")
        group_cols.append("pitcher")

        if ks.empty:
            return pd.DataFrame()

        agg = (
            ks.groupby(group_cols)
            .size()
            .reset_index(name="actual_k")
            .rename(columns={"pitcher": "pitcher_mlbam"})
        )

        if "game_pk" not in agg.columns:
            agg["game_pk"] = 0

        agg["game_pk"]       = agg["game_pk"].astype(int)
        agg["pitcher_mlbam"] = agg["pitcher_mlbam"].astype(int)
        agg["actual_k"]      = agg["actual_k"].astype(float)

        return agg

    @staticmethod
    def _empty_summary(window_days) -> dict:
        return {
            "n_predictions": 0, "n_graded": 0,
            "mae": None, "rmse": None, "bias": None,
            "within_1_k": None, "within_2_k": None,
            "mae_confirmed": None, "mae_projected": None,
            "window_days": window_days, "start_date": "", "end_date": "",
        }


# ── Convenience: auto-record yesterday's actuals ──────────────────────────────

def backfill_actuals(days_back: int = 7):
    """
    Utility to fill in actuals for the last N days in bulk.
    Useful after a gap (e.g. running for the first time after several game days).
    """
    tracker = AccuracyTracker()
    today   = date.today()
    for i in range(1, days_back + 1):
        target = (today - timedelta(days=i)).isoformat()
        tracker.record_actuals(target)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    p = argparse.ArgumentParser(description="MLB Strikeout Accuracy Tracker")
    sub = p.add_subparsers(dest="cmd")

    rec = sub.add_parser("record", help="Fill in actuals for a date")
    rec.add_argument("date", help="YYYY-MM-DD")
    rec.add_argument("--force", action="store_true")

    back = sub.add_parser("backfill", help="Fill actuals for last N days")
    back.add_argument("--days", type=int, default=7)

    summ = sub.add_parser("summary", help="Print performance summary")
    summ.add_argument("--days", type=int, default=None,
                      help="Rolling window in days (omit for all-time)")

    trend = sub.add_parser("trend", help="Print daily MAE table")
    trend.add_argument("--days", type=int, default=30)

    args = p.parse_args()
    tracker = AccuracyTracker()

    if args.cmd == "record":
        n = tracker.record_actuals(args.date, force=args.force)
        print(f"Updated {n} rows for {args.date}")

    elif args.cmd == "backfill":
        backfill_actuals(args.days)

    elif args.cmd == "summary":
        s = tracker.get_summary(window_days=args.days)
        label = f"last {args.days} days" if args.days else "all-time"
        print(f"\n=== Strikeout Model Accuracy ({label}) ===")
        print(f"  Predictions logged : {s['n_predictions']}")
        print(f"  Games graded       : {s['n_graded']}")
        if s["mae"] is not None:
            print(f"  MAE                : {s['mae']:.3f} K")
            print(f"  RMSE               : {s['rmse']:.3f} K")
            print(f"  Bias               : {s['bias']:+.3f} K  ({'over' if s['bias'] > 0 else 'under'}-predicting)")
            print(f"  Within 1 K         : {s['within_1_k']*100:.1f}%")
            print(f"  Within 2 K         : {s['within_2_k']*100:.1f}%")
            if s["mae_confirmed"] is not None:
                print(f"  MAE (confirmed)    : {s['mae_confirmed']:.3f} K")
            if s["mae_projected"] is not None:
                print(f"  MAE (projected)    : {s['mae_projected']:.3f} K")
        else:
            print("  (no graded predictions yet)")

    elif args.cmd == "trend":
        df = tracker.get_daily_mae(window_days=args.days)
        if df.empty:
            print("No graded data yet.")
        else:
            pd.set_option("display.max_rows", 999)
            print(df.to_string(index=False))

    else:
        p.print_help()
