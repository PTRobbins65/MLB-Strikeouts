"""
MLB Strikeout Pipeline — Historical Data Fetcher
Pulls pitch-level Statcast data and season-level FanGraphs stats
via pybaseball. Results are cached to disk as Parquet files.

Usage:
    from data_fetcher import HistoricalDataFetcher
    fetcher = HistoricalDataFetcher()
    pitcher_df = fetcher.get_statcast_pitcher(477132, "2023-04-01", "2023-09-30")
    fg_stats   = fetcher.get_fangraphs_stats(2021, 2024)
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import (
    RAW_DIR, PROCESSED_DIR,
    STATCAST_PITCHER_COLS, FANGRAPHS_PITCHER_COLS,
    WHIFF_DESCRIPTIONS, CSW_DESCRIPTIONS,
    SEASON_START_YEAR,
)

logger = logging.getLogger(__name__)


# ── Lazy imports (pybaseball is slow to import) ───────────────────────────────
def _pybaseball():
    try:
        import pybaseball as pb
        pb.cache.enable()   # disk-caches Statcast requests automatically
        return pb
    except ImportError:
        raise ImportError(
            "pybaseball is required: pip install pybaseball"
        )


class HistoricalDataFetcher:
    """Fetches and caches historical pitching and batting data."""

    def __init__(self, cache_dir: Path = RAW_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Statcast — pitch-level ─────────────────────────────────────────────

    def get_statcast_pitcher(
        self,
        mlbam_id: int,
        start_dt: str,
        end_dt: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return pitch-level Statcast rows for a single pitcher.
        Cached by pitcher × date range.

        Parameters
        ----------
        mlbam_id    : MLB Advanced Media player ID (e.g. 477132 for Kershaw)
        start_dt    : "YYYY-MM-DD"
        end_dt      : "YYYY-MM-DD"
        """
        cache_path = self.cache_dir / f"statcast_pitcher_{mlbam_id}_{start_dt}_{end_dt}.parquet"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached Statcast data: {cache_path.name}")
            return pd.read_parquet(cache_path)

        pb = _pybaseball()
        logger.info(f"Fetching Statcast pitches for mlbam_id={mlbam_id} ({start_dt} -> {end_dt})")
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Statcast fetch timed out for mlbam_id={mlbam_id}")

            # Per-pitcher 90-second hard limit (pybaseball can hang indefinitely)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(90)
            try:
                df = pb.statcast_pitcher(start_dt, end_dt, mlbam_id)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except TimeoutError as exc:
            logger.warning(str(exc))
            return pd.DataFrame()
        except Exception as exc:
            logger.warning(f"Statcast fetch failed for mlbam_id={mlbam_id}: {exc}")
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning(f"No Statcast data returned for mlbam_id={mlbam_id}")
            return pd.DataFrame()

        # Keep only the columns we care about (gracefully ignore missing ones)
        keep = [c for c in STATCAST_PITCHER_COLS if c in df.columns]
        df = df[keep].copy()
        df["game_date"] = pd.to_datetime(df["game_date"])

        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(df):,} rows -> {cache_path.name}")
        return df

    def get_statcast_date_range(
        self,
        start_dt: str,
        end_dt: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return league-wide pitch-level Statcast data for a date range.
        Use this to build the full historical training set.
        Warning: each season is ~700k+ rows.
        """
        cache_path = self.cache_dir / f"statcast_league_{start_dt}_{end_dt}.parquet"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached league Statcast: {cache_path.name}")
            return pd.read_parquet(cache_path)

        pb = _pybaseball()
        logger.info(f"Fetching league-wide Statcast ({start_dt} -> {end_dt}) — may take several minutes")
        df = pb.statcast(start_dt=start_dt, end_dt=end_dt)

        if df is None or df.empty:
            return pd.DataFrame()

        keep = [c for c in STATCAST_PITCHER_COLS if c in df.columns]
        df = df[keep].copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(df):,} rows -> {cache_path.name}")
        return df

    # ── FanGraphs — season-level ───────────────────────────────────────────

    def get_fangraphs_stats(
        self,
        start_season: int,
        end_season: int,
        min_ip: int = 20,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return season-level FanGraphs pitching stats (one row per pitcher-season).
        Columns include K%, SwStr%, CSW%, FIP, xFIP, Stuff+, etc.
        """
        cache_path = self.cache_dir / f"fangraphs_pitching_{start_season}_{end_season}.parquet"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached FanGraphs stats: {cache_path.name}")
            return pd.read_parquet(cache_path)

        pb = _pybaseball()
        logger.info(f"Fetching FanGraphs pitching stats {start_season}–{end_season}")
        try:
            df = pb.pitching_stats(start_season, end_season, qual=min_ip)
        except Exception as exc:
            logger.warning(
                f"FanGraphs pitching stats unavailable ({exc}). "
                "Season-level FanGraphs features will be missing."
            )
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        keep = [c for c in FANGRAPHS_PITCHER_COLS if c in df.columns]
        df = df[keep].copy()
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(df):,} rows -> {cache_path.name}")
        return df

    def get_fangraphs_batter_stats(
        self,
        start_season: int,
        end_season: int,
        min_pa: int = 50,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return season-level FanGraphs batting stats.
        Used to build opponent lineup K% profiles.
        """
        cache_path = self.cache_dir / f"fangraphs_batting_{start_season}_{end_season}.parquet"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached FanGraphs batting: {cache_path.name}")
            return pd.read_parquet(cache_path)

        pb = _pybaseball()
        logger.info(f"Fetching FanGraphs batting stats {start_season}–{end_season}")
        try:
            df = pb.batting_stats(start_season, end_season, qual=min_pa)
        except Exception as exc:
            logger.warning(
                f"FanGraphs batting stats unavailable ({exc}). "
                "Opponent lineup FanGraphs features will be missing."
            )
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # Keep columns relevant to batter K profiling
        desired = [
            "IDfg", "Name", "Season", "Team",
            "PA", "K%", "BB%", "O-Swing%",
            "Contact%", "SwStr%", "wRC+",
            "AVG", "OBP", "SLG",
        ]
        keep = [c for c in desired if c in df.columns]
        df = df[keep].copy()
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(df):,} rows -> {cache_path.name}")
        return df

    # ── Player ID lookup ───────────────────────────────────────────────────

    def lookup_player_id(self, last: str, first: str) -> pd.DataFrame:
        """
        Return a DataFrame with player IDs across systems
        (mlbam, bbref, fangraphs, retro).

        Example
        -------
        lookup_player_id("degrom", "jacob")
        """
        pb = _pybaseball()
        result = pb.playerid_lookup(last, first)
        return result

    # ── Historical lineup reconstruction from Statcast ────────────────────

    def build_lineup_lookup(
        self,
        pitch_df: pd.DataFrame,
        game_meta: Optional[pd.DataFrame] = None,
    ) -> Dict[int, object]:
        """
        Build a game_pk -> LineupCard dict from pitch-level Statcast data.
        Reconstructs batting orders from the batter + at_bat_number columns
        with no additional API calls.

        Parameters
        ----------
        pitch_df  : full Statcast pitch DataFrame (must contain batter,
                    at_bat_number, inning_topbot, game_pk, game_date)
        game_meta : optional DataFrame with game_pk, game_date, home_team,
                    away_team, home_team_id, away_team_id (from game_log)
        """
        from lineup_manager import BatterSlot, LineupCard

        required = {"game_pk", "batter", "at_bat_number", "inning_topbot", "game_date"}
        missing  = required - set(pitch_df.columns)
        if missing:
            logger.warning(
                f"build_lineup_lookup: columns {missing} not in Statcast data. "
                "Re-fetch with --refresh to pull the updated column set."
            )
            return {}

        # Index schedule metadata by game_pk for fast lookup
        meta_index: Dict[int, dict] = {}
        if game_meta is not None and not game_meta.empty:
            for _, row in game_meta.iterrows():
                meta_index[int(row["game_pk"])] = row.to_dict()

        lookup: Dict[int, LineupCard] = {}

        for game_pk, game_pitches in pitch_df.groupby("game_pk"):
            game_pk = int(game_pk)
            meta    = meta_index.get(game_pk, {})

            # "Top" = away team batting (home pitcher on mound)
            # "Bot" = home team batting (away pitcher on mound)
            away_batters = self._extract_batting_order(
                game_pitches[game_pitches["inning_topbot"] == "Top"]
            )
            home_batters = self._extract_batting_order(
                game_pitches[game_pitches["inning_topbot"] == "Bot"]
            )

            def _safe_int(val, default=0):
                try:
                    return int(val) if val == val else default  # val != val is True for NaN
                except (TypeError, ValueError):
                    return default

            lookup[game_pk] = LineupCard(
                game_pk      = game_pk,
                game_date    = str(game_pitches["game_date"].iloc[0])[:10],
                home_team    = str(meta.get("home_team", "") or ""),
                away_team    = str(meta.get("away_team", "") or ""),
                home_team_id = _safe_int(meta.get("home_team_id", 0)),
                away_team_id = _safe_int(meta.get("away_team_id", 0)),
                home_batters = home_batters,
                away_batters = away_batters,
                confirmed    = True,
            )

        logger.info(
            f"Lineup lookup built: {len(lookup):,} games, "
            f"avg {sum(len(c.home_batters) for c in lookup.values()) / max(len(lookup), 1):.1f} "
            f"home batters per game"
        )
        return lookup

    @staticmethod
    def _extract_batting_order(half_pitches: pd.DataFrame, max_batters: int = 9) -> list:
        """
        Return BatterSlot list ordered by first at_bat_number appearance.
        Works for one half-inning side (all Top or all Bot pitches for a game).
        """
        from lineup_manager import BatterSlot

        if half_pitches.empty or "at_bat_number" not in half_pitches.columns:
            return []

        first_ab = (
            half_pitches.groupby("batter")["at_bat_number"]
            .min()
            .sort_values()
            .head(max_batters)
            .reset_index()
        )

        return [
            BatterSlot(
                batting_order = slot + 1,
                player_id     = int(row["batter"]),
                full_name     = "Unknown",   # name not needed for feature building
                position      = "?",
            )
            for slot, (_, row) in enumerate(first_ab.iterrows())
        ]

    # ── Computed per-start metrics from Statcast ───────────────────────────

    def compute_per_start_metrics(self, pitch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate pitch-level Statcast rows into one row per start.

        Returns columns:
            game_date, pitcher, player_name,
            strikeouts, pitches, innings_pitched,
            whiff_pct, csw_pct, zone_pct, o_swing_pct,
            avg_velo, avg_spin, avg_pfx_x, avg_pfx_z,
            pitch_type_pcts  (one column per pitch type, e.g. FF_pct, SL_pct)
        """
        if pitch_df.empty:
            return pd.DataFrame()

        df = pitch_df.copy()

        # ── Core per-pitch flags ──────────────────────────────────────────
        df["is_whiff"]         = df["description"].isin(WHIFF_DESCRIPTIONS)
        df["is_csw"]           = df["description"].isin(CSW_DESCRIPTIONS)
        df["is_in_zone"]       = df["zone"].between(1, 9, inclusive="both")
        df["is_swing"]         = df["description"].str.contains("swing|foul", na=False)
        df["is_out_of_zone_swing"] = (~df["is_in_zone"]) & df["is_swing"]
        df["is_strikeout"]     = df["events"] == "strikeout"

        # ── Group by game ─────────────────────────────────────────────────
        # Include game_pk when available for accurate per-game grouping
        base_keys = ["game_date", "pitcher", "player_name"]
        group_keys = (["game_pk"] + base_keys) if "game_pk" in df.columns else base_keys

        grp = df.groupby(group_keys)

        agg = grp.agg(
            pitches          = ("pitch_type", "count"),
            strikeouts       = ("is_strikeout", "sum"),
            whiffs           = ("is_whiff", "sum"),
            csw_pitches      = ("is_csw", "sum"),
            swings           = ("is_swing", "sum"),
            in_zone_pitches  = ("is_in_zone", "sum"),
            out_of_zone_swings = ("is_out_of_zone_swing", "sum"),
            out_of_zone_pitches = ("is_in_zone", lambda x: (~x).sum()),
            avg_velo         = ("release_speed", "mean"),
            avg_spin         = ("release_spin_rate", "mean"),
            avg_pfx_x        = ("pfx_x", "mean"),
            avg_pfx_z        = ("pfx_z", "mean"),
            avg_extension    = ("release_extension", "mean"),
        ).reset_index()

        # ── Rate stats ────────────────────────────────────────────────────
        agg["whiff_pct"]   = agg["whiffs"]  / agg["swings"].clip(lower=1)
        agg["csw_pct"]     = agg["csw_pitches"] / agg["pitches"].clip(lower=1)
        agg["zone_pct"]    = agg["in_zone_pitches"] / agg["pitches"].clip(lower=1)
        agg["o_swing_pct"] = agg["out_of_zone_swings"] / agg["out_of_zone_pitches"].clip(lower=1)

        # ── is_home: "Top" half = away batting = home team pitching ──────
        if "inning_topbot" in df.columns:
            is_home_df = (
                df.groupby(group_keys)["inning_topbot"]
                .agg(lambda x: (x == "Top").mean() > 0.5)
                .reset_index()
                .rename(columns={"inning_topbot": "is_home"})
            )
            agg = agg.merge(is_home_df, on=group_keys, how="left")

        # ── Pitch mix (% usage per type) ──────────────────────────────────
        mix_keys = (["game_pk", "game_date", "pitcher"] if "game_pk" in df.columns
                    else ["game_date", "pitcher"])
        pitch_mix = (
            df.groupby(mix_keys + ["pitch_type"])
              .size()
              .unstack(fill_value=0)
        )
        pitch_mix = pitch_mix.div(pitch_mix.sum(axis=1), axis=0)
        pitch_mix.columns = [f"pitch_pct_{c}" for c in pitch_mix.columns]
        pitch_mix = pitch_mix.reset_index()

        agg = agg.merge(pitch_mix, on=mix_keys, how="left")
        agg["game_date"] = pd.to_datetime(agg["game_date"])
        agg = agg.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

        return agg
