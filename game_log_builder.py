"""
MLB Strikeout Pipeline — Game Log Builder
Assembles the historical training game log: one row per pitcher start,
with actual strikeout total and game context (park, home/away, opponent).

Joins:
  - Statcast per-start metrics  (game_pk, strikeouts, pitcher_hand, is_home)
  - MLB StatsAPI season schedule (game_pk -> park_id, team names, is_night_game)

Output: data/processed/game_log_{start_year}_{end_year}.parquet

Usage
-----
    python game_log_builder.py --start 2021 --end 2024
    python game_log_builder.py --start 2021 --end 2024 --refresh
"""

import argparse
import logging
import sys
from typing import List

import pandas as pd
import requests

from config import MLB_API_BASE, PROCESSED_DIR, LOG_DIR
from data_fetcher import HistoricalDataFetcher

logger = logging.getLogger(__name__)

OUTPUT_COLS = [
    "game_pk", "game_date", "pitcher_mlbam", "pitcher_hand",
    "is_home", "park_id", "is_night_game",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "strikeouts",
]


def _mlb_get(endpoint: str, params: dict = None) -> dict:
    url = f"{MLB_API_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


class GameLogBuilder:
    """
    Builds (or loads cached) the historical pitcher-start game log.
    Output schema matches what FeatureBuilder.build_training_set() expects.
    """

    def __init__(self):
        self.fetcher = HistoricalDataFetcher()

    def build(
        self,
        start_year: int,
        end_year: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = PROCESSED_DIR / f"game_log_{start_year}_{end_year}.parquet"
        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached game log: {cache_path.name}")
            return pd.read_parquet(cache_path)

        seasons = []
        for year in range(start_year, end_year + 1):
            logger.info(f"Building game log for {year}...")
            df = self._build_season(year)
            if not df.empty:
                seasons.append(df)

        if not seasons:
            logger.warning("No data assembled — returning empty game log")
            return pd.DataFrame()

        game_log = pd.concat(seasons, ignore_index=True)
        game_log.to_parquet(cache_path, index=False)
        logger.info(
            f"Game log saved: {len(game_log):,} pitcher-starts "
            f"({start_year}–{end_year}) -> {cache_path.name}"
        )
        return game_log

    # ── Per-season build ───────────────────────────────────────────────────

    def _build_season(self, year: int) -> pd.DataFrame:
        start_dt = f"{year}-03-01"
        end_dt   = f"{year}-11-01"

        # ── 1. Statcast pitch-level data ───────────────────────────────────
        pitches = self.fetcher.get_statcast_date_range(start_dt, end_dt)
        if pitches.empty:
            logger.warning(f"No Statcast data for {year}")
            return pd.DataFrame()

        # ── 2. Per-start aggregation ───────────────────────────────────────
        starts = self.fetcher.compute_per_start_metrics(pitches)
        if starts.empty:
            return pd.DataFrame()

        # Pitcher handedness from p_throws (mode per pitcher)
        if "p_throws" in pitches.columns:
            hand_by_pitcher = (
                pitches.groupby("pitcher")["p_throws"]
                .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "R")
                .reset_index()
                .rename(columns={"pitcher": "pitcher_mlbam", "p_throws": "pitcher_hand"})
            )
        else:
            hand_by_pitcher = None

        starts = starts.rename(columns={"pitcher": "pitcher_mlbam"})
        starts["game_date"] = pd.to_datetime(starts["game_date"]).dt.date

        if hand_by_pitcher is not None:
            starts = starts.merge(hand_by_pitcher, on="pitcher_mlbam", how="left")
            starts["pitcher_hand"] = starts["pitcher_hand"].fillna("R")
        else:
            starts["pitcher_hand"] = "R"

        # ── 3. Schedule metadata ───────────────────────────────────────────
        schedule = self._fetch_season_schedule(year)

        if not schedule.empty and "game_pk" in starts.columns:
            # Drop game_date from schedule before merging — starts already has it
            # from Statcast data; keeping both creates game_date_x / game_date_y.
            sched_cols = [c for c in schedule.columns if c != "game_date"]
            starts = starts.merge(schedule[sched_cols], on="game_pk", how="left")
        elif not schedule.empty:
            # Fall back to date-based join if game_pk not in Statcast output
            starts = starts.merge(schedule, on="game_date", how="left")

        # Fill defaults for any missing context columns
        defaults = {
            "park_id": 680, "is_home": True, "is_night_game": True,
            "game_pk": 0, "home_team": "", "away_team": "",
            "home_team_id": 0, "away_team_id": 0,
        }
        for col, val in defaults.items():
            if col not in starts.columns:
                starts[col] = val

        present = [c for c in OUTPUT_COLS if c in starts.columns]
        return starts[present].dropna(subset=["strikeouts"]).copy()

    # ── StatsAPI schedule fetch ────────────────────────────────────────────

    def _fetch_season_schedule(self, year: int) -> pd.DataFrame:
        logger.info(f"Fetching {year} schedule from StatsAPI...")
        try:
            data = _mlb_get("schedule", params={
                "sportId":  1,
                "season":   year,
                "gameType": "R",
                "hydrate":  "team,venue",
                "fields": (
                    "dates,games,gamePk,status,abstractGameState,gameDate,"
                    "teams,away,home,team,id,name,venue,id"
                ),
            })
        except Exception as exc:
            logger.error(f"Schedule fetch failed for {year}: {exc}")
            return pd.DataFrame()

        rows = []
        for date_entry in data.get("dates", []):
            for g in date_entry.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                game_dt_str = g.get("gameDate", "")
                try:
                    ts        = pd.Timestamp(game_dt_str)
                    game_date = ts.date()
                    # Games starting after 18:00 UTC (≈ 2 PM ET) counted as night
                    is_night  = ts.hour >= 18
                except Exception:
                    continue

                away = g["teams"]["away"]
                home = g["teams"]["home"]
                rows.append({
                    "game_pk":      g["gamePk"],
                    "game_date":    game_date,
                    "park_id":      g.get("venue", {}).get("id", 680),
                    "home_team":    home["team"]["name"],
                    "away_team":    away["team"]["name"],
                    "home_team_id": home["team"]["id"],
                    "away_team_id": away["team"]["id"],
                    "is_night_game": is_night,
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "game_log_builder.log"),
        ],
    )
    p = argparse.ArgumentParser(description="Build historical MLB pitcher-start game log")
    p.add_argument("--start",   type=int, default=2021, help="First season (default 2021)")
    p.add_argument("--end",     type=int, default=2024, help="Last season (default 2024)")
    p.add_argument("--refresh", action="store_true",    help="Re-fetch all data")
    args = p.parse_args()

    builder  = GameLogBuilder()
    game_log = builder.build(args.start, args.end, force_refresh=args.refresh)

    if game_log.empty:
        print("No data assembled — check logs for errors")
    else:
        print(f"\nGame log: {len(game_log):,} pitcher-starts ({args.start}–{args.end})")
        print(game_log.head(10).to_string(index=False))
