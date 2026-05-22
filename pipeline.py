"""
MLB Strikeout Pipeline — Daily Orchestrator
Runs the full end-to-end pipeline for a given date:

  1. Fetch today's schedule + probable pitchers
  2. Build projected lineups from recent history (pre-lineup-lock)
  3. Start background polling to detect confirmed lineup releases
  4. Assemble feature rows for each starting pitcher
  5. Emit a predictions DataFrame (once a model is attached)

Run modes
---------
  python pipeline.py                  # today's games
  python pipeline.py --date 2024-06-15
  python pipeline.py --historical 2023  # build training data for a full season

Architecture
------------
  pipeline.py
    ├── lineup_manager.py   ← Phase 1 (projected) + Phase 2 (confirmed polling)
    ├── data_fetcher.py     ← Statcast + FanGraphs historical data
    ├── feature_builder.py  ← feature engineering
    └── config.py           ← constants
"""

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config import DATA_DIR, FEATURES_DIR, LOG_DIR, MODEL_DIR, ROLLING_WINDOWS
from data_fetcher import HistoricalDataFetcher
from feature_builder import FeatureBuilder
from lineup_manager import LineupManager
from model_trainer import FEATURE_COLS

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")


class DailyPipeline:
    """
    Orchestrates one full pipeline run for a single game date.

    Parameters
    ----------
    target_date     : date to run (defaults to today)
    start_seasons   : how far back to pull historical data (default: 3 years)
    """

    def __init__(
        self,
        target_date: Optional[date] = None,
        history_years: int = 3,
    ):
        self.target_date   = target_date or date.today()
        self.history_years = history_years

        self.fetcher  = HistoricalDataFetcher()
        self.lineup   = LineupManager(target_date=self.target_date)
        self.builder: Optional[FeatureBuilder] = None   # built after data load

    # ── Step 1: Load historical data ───────────────────────────────────────

    def load_historical_data(self):
        """Pull FanGraphs pitcher + batter stats for the lookback window."""
        current_year = self.target_date.year
        start_year   = current_year - self.history_years

        logger.info(f"Loading FanGraphs data {start_year}–{current_year}")
        self.fg_pitchers = self.fetcher.get_fangraphs_stats(start_year, current_year)
        self.fg_batters  = self.fetcher.get_fangraphs_batter_stats(start_year, current_year)

        logger.info(
            f"FanGraphs loaded: {len(self.fg_pitchers):,} pitcher-seasons, "
            f"{len(self.fg_batters):,} batter-seasons"
        )

    # ── Step 2: Fetch today's schedule ────────────────────────────────────

    def fetch_schedule(self) -> List[dict]:
        logger.info(f"Fetching schedule for {self.target_date}")
        games = self.lineup.get_today_games()
        logger.info(f"{len(games)} games scheduled")
        return games

    # ── Step 3: Build projected lineups ───────────────────────────────────

    def build_projected_lineups(self, games: List[dict]):
        """
        For games where confirmed lineups aren't yet published,
        build historical-frequency projections.
        """
        for game in games:
            card = self.lineup.get_lineup(game["game_pk"])
            if card is None:
                continue
            if card.confirmed:
                logger.info(f"game_pk={game['game_pk']}: lineup already confirmed")
                continue

            # Project away lineup
            pp_home = game.get("probable_pitcher_home") or {}
            pp_away = game.get("probable_pitcher_away") or {}

            home_throws = pp_home.get("throws", "R")
            away_throws = pp_away.get("throws", "R")

            projected_away = self.lineup.build_projected_lineup(
                team_id       = game["away_team_id"],
                game_pk       = game["game_pk"],
                pitcher_throws= home_throws,   # away batters face home pitcher
            )
            projected_home = self.lineup.build_projected_lineup(
                team_id       = game["home_team_id"],
                game_pk       = game["game_pk"],
                pitcher_throws= away_throws,
            )

            with self.lineup._lock:
                card.away_batters = projected_away or card.away_batters
                card.home_batters = projected_home or card.home_batters

            logger.info(
                f"game_pk={game['game_pk']}: projected "
                f"{len(projected_away)} away + {len(projected_home)} home batters"
            )

    # ── Step 4: Load pitcher Statcast histories ────────────────────────────

    def load_pitcher_statcast(self, games: List[dict]) -> pd.DataFrame:
        """
        For each probable pitcher in today's games, fetch their Statcast
        pitch history and compute per-start metrics.
        Returns a single DataFrame with all pitchers' start-level stats.
        """
        all_starts = []
        seen_ids   = set()

        start_dt = (self.target_date - timedelta(days=365 * self.history_years)).strftime("%Y-%m-%d")
        end_dt   = (self.target_date - timedelta(days=1)).strftime("%Y-%m-%d")

        for game in games:
            for side in ["probable_pitcher_home", "probable_pitcher_away"]:
                pp = game.get(side)
                if not pp or pp.get("id") is None:
                    continue
                pid = pp["id"]
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)

                logger.info(f"Loading Statcast for {pp['fullName']} (mlbam={pid})")
                pitches = self.fetcher.get_statcast_pitcher(pid, start_dt, end_dt)
                if pitches.empty:
                    logger.warning(f"No Statcast data for {pp['fullName']}")
                    continue

                starts = self.fetcher.compute_per_start_metrics(pitches)
                all_starts.append(starts)

        if not all_starts:
            logger.warning("No per-start Statcast data assembled")
            return pd.DataFrame()

        combined = pd.concat(all_starts, ignore_index=True)
        logger.info(f"Per-start data: {len(combined):,} rows for {len(seen_ids)} pitchers")
        return combined

    # ── Step 5: Assemble feature rows ─────────────────────────────────────

    def assemble_features(
        self,
        games: List[dict],
        statcast_starts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build one feature row per starting pitcher per game.
        Returns a DataFrame ready for model inference.
        """
        if self.fg_pitchers is None or self.fg_batters is None:
            raise RuntimeError("Call load_historical_data() first")

        self.builder = FeatureBuilder(
            fg_pitcher_df   = self.fg_pitchers,
            fg_batter_df    = self.fg_batters,
            statcast_starts = statcast_starts,
        )

        feature_rows = []
        date_str = self.target_date.strftime("%Y-%m-%d")

        for game in games:
            card = self.lineup.get_lineup(game["game_pk"])
            if card is None:
                continue

            for side in ["home", "away"]:
                pp_key   = f"probable_pitcher_{side}"
                pp       = game.get(pp_key)
                if not pp or pp.get("id") is None:
                    continue

                is_home  = (side == "home")
                row = self.builder.build_row(
                    pitcher_mlbam_id = pp["id"],
                    game_pk          = game["game_pk"],
                    game_date        = date_str,
                    lineup_card      = card,
                    pitcher_hand     = pp.get("throws", "R"),
                    is_home          = is_home,
                    park_id          = game.get("venue_id", 680),
                )

                if row is None:
                    logger.warning(
                        f"Feature build failed for {pp['fullName']} "
                        f"game_pk={game['game_pk']}"
                    )
                    continue

                row["pitcher_name"] = pp["fullName"]
                row["home_team"]    = game["home_team"]
                row["away_team"]    = game["away_team"]
                feature_rows.append(row)

        df = pd.DataFrame(feature_rows)
        logger.info(f"Assembled {len(df)} pitcher feature rows")
        return df

    # ── Full daily run ─────────────────────────────────────────────────────

    def run(self, wait_for_lineups: bool = True) -> pd.DataFrame:
        """
        Execute all pipeline steps and return the feature DataFrame.

        If wait_for_lineups=True, the pipeline starts the background poller
        and waits until all lineups are confirmed (or the cutoff time passes)
        before building features.
        """
        logger.info(f"=== Daily Pipeline Run: {self.target_date} ===")

        # 1. Historical data
        self.load_historical_data()

        # 2. Today's schedule
        games = self.fetch_schedule()
        if not games:
            logger.info("No games today — pipeline complete")
            return pd.DataFrame()

        # 3. Projected lineups
        self.build_projected_lineups(games)

        # 4. Pitcher Statcast history
        statcast_starts = self.load_pitcher_statcast(games)

        # 5. Start polling for confirmed lineups
        if wait_for_lineups:
            self.lineup.start_polling()
            logger.info("Polling for confirmed lineups — waiting up to 4 hours before first pitch...")
            # In production this loop runs continuously; here we do a single
            # blocking check that can be interrupted by Ctrl+C or a scheduler
            try:
                self._wait_for_confirmations(games, max_wait_minutes=240)
            except KeyboardInterrupt:
                logger.info("Interrupted — proceeding with available lineups")
            finally:
                self.lineup.stop_polling()

        # 6. Feature assembly
        features_df = self.assemble_features(games, statcast_starts)

        # 7. Persist to disk
        if not features_df.empty:
            out_path = FEATURES_DIR / f"features_{self.target_date}.parquet"
            features_df.to_parquet(out_path, index=False)
            logger.info(f"Features saved → {out_path}")

        # 8. Predictions (if a trained model exists)
        features_df = self._add_predictions(features_df)

        return features_df

    def _add_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Load the saved model and append a predicted_k column."""
        if features_df.empty:
            return features_df

        xgb_path = MODEL_DIR / "strikeout_xgb.json"
        glm_path  = MODEL_DIR / "strikeout_glm.joblib"

        model = None
        model_name = ""

        if xgb_path.exists():
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(str(xgb_path))
                model_name = "XGBoost"
            except Exception as exc:
                logger.warning(f"Could not load XGBoost model: {exc}")

        if model is None and glm_path.exists():
            try:
                import joblib
                model = joblib.load(glm_path)
                model_name = "Poisson GLM"
            except Exception as exc:
                logger.warning(f"Could not load Poisson GLM: {exc}")

        if model is None:
            logger.info("No trained model found — skipping predictions (run model_trainer.py first)")
            return features_df

        avail = [c for c in FEATURE_COLS if c in features_df.columns]
        X = features_df[avail].copy()
        preds = np.clip(model.predict(X), 0, None)
        features_df = features_df.copy()
        features_df["predicted_k"] = preds.round(1)
        logger.info(f"Predictions added using {model_name}")
        return features_df

    def _wait_for_confirmations(self, games: List[dict], max_wait_minutes: int = 240):
        """
        Block until all games have confirmed lineups, or the timeout is reached.
        In a scheduler environment you'd replace this with a proper async approach.
        """
        deadline = datetime.now() + timedelta(minutes=max_wait_minutes)
        while datetime.now() < deadline:
            all_confirmed = all(
                (self.lineup.get_lineup(g["game_pk"]) or type("", (), {"confirmed": True})()).confirmed
                for g in games
            )
            if all_confirmed:
                logger.info("All lineups confirmed!")
                return
            remaining = (deadline - datetime.now()).seconds // 60
            logger.info(f"Waiting for lineups — {remaining} min remaining until timeout")
            time.sleep(60)

        unconfirmed = [
            g["away_team"] + " @ " + g["home_team"]
            for g in games
            if not (self.lineup.get_lineup(g["game_pk"]) or
                    type("", (), {"confirmed": False})()).confirmed
        ]
        if unconfirmed:
            logger.warning(
                f"Timeout reached. Proceeding with projected lineups for: "
                + ", ".join(unconfirmed)
            )


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MLB Strikeout Prediction Pipeline")
    p.add_argument("--date",       type=str, help="YYYY-MM-DD  (default: today)")
    p.add_argument("--no-wait",    action="store_true", help="Skip lineup polling wait")
    p.add_argument("--history",    type=int, default=3, help="Years of historical data")
    p.add_argument("--show",       action="store_true", help="Print feature table to stdout")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_date = date.fromisoformat(args.date) if args.date else date.today()

    pipeline = DailyPipeline(target_date=run_date, history_years=args.history)
    features = pipeline.run(wait_for_lineups=not args.no_wait)

    if args.show and not features.empty:
        pd.set_option("display.max_columns", 12)
        pd.set_option("display.width", 140)
        print("\n── Today's Strikeout Predictions ──")
        show_cols = [c for c in [
            "pitcher_name", "home_team", "away_team",
            "k_rolling_5", "whiff_pct_5", "opp_lineup_k_pct",
            "park_k_factor", "lineup_confirmed", "predicted_k",
        ] if c in features.columns]
        print(features[show_cols].to_string(index=False))
