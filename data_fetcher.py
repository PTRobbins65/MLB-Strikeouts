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

        Cache strategy (incremental):
        - Cache file: statcast_pitcher_{mlbam_id}.parquet  (no date range in name)
        - On hit: if cached data covers through yesterday, return it immediately
        - If stale: fetch only the delta (last cached date → end_dt) and append
        - First run: full fetch from start_dt → end_dt, then cached forever

        This means after the first run each pitcher costs ~1 small API call per
        day instead of a full re-download.

        Parameters
        ----------
        mlbam_id : MLB Advanced Media player ID
        start_dt : "YYYY-MM-DD" — earliest date needed
        end_dt   : "YYYY-MM-DD" — latest date needed (usually yesterday)
        """
        cache_path = self.cache_dir / f"statcast_pitcher_v2_{mlbam_id}.parquet"
        start_date = pd.to_datetime(start_dt)
        end_date   = pd.to_datetime(end_dt)

        cached_df = pd.DataFrame()
        fetch_from = start_date   # default: full fetch

        if cache_path.exists() and not force_refresh:
            try:
                cached_df  = pd.read_parquet(cache_path)
                cached_df["game_date"] = pd.to_datetime(cached_df["game_date"])
                max_cached = cached_df["game_date"].max()

                if max_cached >= end_date - timedelta(days=1):
                    # Cache is fresh — filter to requested window and return
                    logger.info(
                        f"Cache hit for mlbam_id={mlbam_id} "
                        f"(cached through {max_cached.date()})"
                    )
                    return cached_df[cached_df["game_date"] >= start_date].reset_index(drop=True)

                # Cache exists but is stale — only fetch the delta
                fetch_from = max_cached + timedelta(days=1)
                logger.info(
                    f"Incremental update for mlbam_id={mlbam_id}: "
                    f"{fetch_from.date()} → {end_dt}"
                )
            except Exception as exc:
                logger.warning(f"Cache read failed for mlbam_id={mlbam_id}: {exc} — full refresh")
                cached_df  = pd.DataFrame()
                fetch_from = start_date

        fetch_start_str = fetch_from.strftime("%Y-%m-%d")

        pb = _pybaseball()
        logger.info(
            f"Fetching Statcast for mlbam_id={mlbam_id} "
            f"({fetch_start_str} → {end_dt})"
        )
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Statcast fetch timed out for mlbam_id={mlbam_id}")

            # Per-pitcher 90-second hard limit (pybaseball can hang indefinitely)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(90)
            try:
                new_df = pb.statcast_pitcher(fetch_start_str, end_dt, mlbam_id)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except TimeoutError as exc:
            logger.warning(str(exc))
            # Return whatever we have in cache rather than nothing
            return cached_df[cached_df["game_date"] >= start_date].reset_index(drop=True) if not cached_df.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning(f"Statcast fetch failed for mlbam_id={mlbam_id}: {exc}")
            return cached_df[cached_df["game_date"] >= start_date].reset_index(drop=True) if not cached_df.empty else pd.DataFrame()

        if new_df is None or new_df.empty:
            logger.info(f"No new Statcast rows for mlbam_id={mlbam_id} (up to date)")
            return cached_df[cached_df["game_date"] >= start_date].reset_index(drop=True) if not cached_df.empty else pd.DataFrame()

        # Trim to requested columns
        keep   = [c for c in STATCAST_PITCHER_COLS if c in new_df.columns]
        new_df = new_df[keep].copy()
        new_df["game_date"] = pd.to_datetime(new_df["game_date"])

        # Merge with existing cache — no dedup needed because incremental fetch
        # starts from max_cached_date+1, so there is no overlap between the two.
        # Sorting by game_date keeps the file in chronological order.
        combined = pd.concat([cached_df, new_df], ignore_index=True)
        combined = combined.sort_values(["game_date", "at_bat_number"]).reset_index(drop=True)

        combined.to_parquet(cache_path, index=False)
        logger.info(
            f"Cache updated for mlbam_id={mlbam_id}: "
            f"{len(new_df):,} new rows, {len(combined):,} total → {cache_path.name}"
        )

        return combined[combined["game_date"] >= start_date].reset_index(drop=True)

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

    # ── Computed season-level stats from Statcast ─────────────────────────

    def compute_pitcher_season_stats(self, pitches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw pitch-level Statcast data into one row per (pitcher, season).

        Replaces the FanGraphs season-level stats that are blocked (403).  Uses
        only columns already in the Statcast cache, so no extra API calls are
        needed.

        Columns returned
        ----------------
        pitcher, season,
        k_pct_season    — strikeouts / batters faced
        bb_pct_season   — walks / batters faced
        swstr_season    — whiffs / total pitches (SwStr%)
        k9_season       — K per 9 estimated innings (PA / 4.2 ≈ IP)
        fip_season      — FIP using estimated IP + FIP constant 3.17
        avg_velo_season — mean release speed
        n_pa            — batters faced (quality filter; exclude low-sample seasons)
        """
        if pitches_df.empty:
            return pd.DataFrame()

        df = pitches_df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["season"] = df["game_date"].dt.year

        # Event flags — events column is only populated on the last pitch of each PA
        df["is_k"]   = df["events"] == "strikeout"
        df["is_bb"]  = df["events"].isin(["walk", "intent_walk"])
        df["is_hr"]  = df["events"] == "home_run"
        df["is_hbp"] = df["events"] == "hit_by_pitch"
        # Any non-null events value = end of a plate appearance
        df["is_pa"]  = df["events"].notna() & (df["events"].astype(str).str.strip() != "")
        df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)

        grp = df.groupby(["pitcher", "season"])

        agg = grp.agg(
            n_pa      = ("is_pa",    "sum"),
            k         = ("is_k",     "sum"),
            bb        = ("is_bb",    "sum"),
            hr        = ("is_hr",    "sum"),
            hbp       = ("is_hbp",   "sum"),
            n_pitches = ("pitch_type", "count"),
            whiffs    = ("is_whiff", "sum"),
            avg_velo  = ("release_speed", "mean"),
        ).reset_index()

        pa  = agg["n_pa"].clip(lower=1)
        ip  = pa / 4.2           # estimate: ~4.2 batters faced per inning (MLB avg)

        agg["k_pct_season"]    = agg["k"]      / pa
        agg["bb_pct_season"]   = agg["bb"]     / pa
        agg["swstr_season"]    = agg["whiffs"] / agg["n_pitches"].clip(lower=1)
        agg["k9_season"]       = agg["k"]      / ip * 9
        agg["avg_velo_season"] = agg["avg_velo"]

        # FIP = (13·HR + 3·(BB+HBP) − 2·K) / IP + constant
        # FIP constant ≈ 3.17 (recent MLB average ERA − raw FIP)
        FIP_CONSTANT = 3.17
        agg["fip_season"] = (
            (13 * agg["hr"] + 3 * (agg["bb"] + agg["hbp"]) - 2 * agg["k"])
            / ip
            + FIP_CONSTANT
        )

        # Filter to pitchers with at least 50 PA (avoids noise from spot appearances)
        agg = agg[agg["n_pa"] >= 50].copy()

        keep = [
            "pitcher", "season",
            "k_pct_season", "bb_pct_season", "swstr_season",
            "k9_season", "fip_season", "avg_velo_season", "n_pa",
        ]
        result = agg[keep].reset_index(drop=True)
        logger.info(
            f"Season stats computed: {len(result)} pitcher-seasons "
            f"({result['season'].nunique()} seasons, "
            f"{result['pitcher'].nunique()} pitchers)"
        )
        return result

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
