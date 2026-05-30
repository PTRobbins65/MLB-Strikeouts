"""
MLB Strikeout Pipeline — StatsAPI Data Fetcher

Pulls all *counting* and *rolling* data from the official MLB StatsAPI
(statsapi.mlb.com), which — unlike FanGraphs (403) and Baseball Savant
(fragile per-pitcher scraping) — is fast, unauthenticated, and reliable.

What this module provides
-------------------------
  get_pitcher_season_stats(season)   one bulk call -> every pitcher's
                                     k_pct_season, bb_pct_season, k9_season,
                                     fip_season (real IP/BF/HR/HBP, not estimated)
  get_batter_season_stats(season)    one bulk call -> every batter's
                                     k_pct, bb_pct
  get_pitcher_gamelog(pid, seasons)  per-start K / pitches / IP / date / opp,
                                     used for k_rolling_*, days_rest, pitches_last,
                                     and the strikeouts target

What it intentionally does NOT provide
--------------------------------------
  Pitch-level "stuff" (whiff%, csw%, zone%, o_swing%, avg_velo) and batter
  plate-discipline are physically Statcast-only — those come from the daily
  league-wide snapshot (see daily_snapshot.py), not from here.

Usage
-----
    from statsapi_fetcher import StatsAPIFetcher
    f = StatsAPIFetcher()
    pit = f.get_pitcher_season_stats(2025)
    bat = f.get_batter_season_stats(2025)
    log = f.get_pitcher_gamelog(594798, [2024, 2025])
"""

import logging
from typing import Iterable, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import MLB_API_BASE, RAW_DIR

logger = logging.getLogger(__name__)

# FIP constant ~ recent MLB average (ERA - raw FIP). Same value used in the
# old Statcast-derived stats so model features stay on a comparable scale.
FIP_CONSTANT = 3.17


def _splits(data: dict) -> list:
    """
    Safely extract the 'splits' list from a StatsAPI stats response.

    The payload shape is {"stats": [{"splits": [...]}]}, but 'stats' can come
    back as an empty list (player/season with no data), so a bare [0] index
    raises IndexError. This returns [] in every empty/malformed case.
    """
    stats = data.get("stats") or []
    if not stats:
        return []
    return stats[0].get("splits", []) or []


def _parse_ip(value) -> float:
    """
    Convert StatsAPI innings-pitched notation to decimal innings.
    "180.1" -> 180 + 1/3, "10.2" -> 10 + 2/3, "5.0" -> 5.0
    """
    if value is None:
        return 0.0
    try:
        whole_str, _, frac_str = str(value).partition(".")
        whole = int(whole_str) if whole_str else 0
        outs = int(frac_str[0]) if frac_str else 0   # the digit after '.' = outs (0-2)
        return whole + outs / 3.0
    except (ValueError, TypeError):
        return 0.0


class StatsAPIFetcher:
    """Fetches season-level and per-start stats from the MLB StatsAPI."""

    def __init__(self, cache_dir=RAW_DIR, timeout: int = 20):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        """Session with retry/backoff so transient 5xx / rate limits self-heal."""
        s = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=0.6,           # 0.6, 1.2, 2.4, 4.8s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "mlb-strikeout-pipeline/1.0"})
        return s

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{MLB_API_BASE}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ── Pitcher season stats (bulk, 1 call per season) ────────────────────

    def get_pitcher_season_stats(
        self,
        season: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return one row per pitcher for `season` with the FanGraphs-replacement
        season features: k_pct_season, bb_pct_season, k9_season, fip_season.

        Columns: pitcher (mlbam id), season, n_pa (battersFaced),
                 k_pct_season, bb_pct_season, k9_season, fip_season
        """
        cache_path = self.cache_dir / f"statsapi_pitcher_season_{season}.parquet"
        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached StatsAPI pitcher season: {cache_path.name}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching StatsAPI pitcher season stats {season} (bulk)")
        try:
            data = self._get("stats", {
                "stats": "season", "group": "pitching", "season": season,
                "playerPool": "all", "limit": 3000, "gameType": "R",
            })
        except Exception as exc:
            logger.warning(f"StatsAPI pitcher season fetch failed for {season}: {exc}")
            return pd.DataFrame()

        rows = []
        for split in _splits(data):
            pid = split.get("player", {}).get("id")
            st = split.get("stat", {})
            if pid is None:
                continue
            bf = st.get("battersFaced", 0) or 0
            k  = st.get("strikeOuts", 0) or 0
            bb = st.get("baseOnBalls", 0) or 0
            hr = st.get("homeRuns", 0) or 0
            hbp = st.get("hitByPitch", 0) or 0
            ip = _parse_ip(st.get("inningsPitched"))
            rows.append({
                "pitcher": int(pid), "season": season,
                "n_pa": int(bf), "k": int(k), "bb": int(bb),
                "hr": int(hr), "hbp": int(hbp), "ip": ip,
            })

        if not rows:
            logger.warning(f"No pitcher season rows returned for {season}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        pa = df["n_pa"].clip(lower=1)
        ip = df["ip"].clip(lower=0.1)
        df["k_pct_season"]  = df["k"] / pa
        df["bb_pct_season"] = df["bb"] / pa
        df["k9_season"]     = df["k"] / ip * 9
        df["fip_season"]    = (
            (13 * df["hr"] + 3 * (df["bb"] + df["hbp"]) - 2 * df["k"]) / ip
            + FIP_CONSTANT
        )

        # Quality filter: drop spot relievers / tiny samples (mirrors prior >=50 BF)
        df = df[df["n_pa"] >= 50].copy()

        keep = ["pitcher", "season", "k_pct_season", "bb_pct_season",
                "k9_season", "fip_season", "n_pa"]
        result = df[keep].reset_index(drop=True)
        result.to_parquet(cache_path, index=False)
        logger.info(f"Pitcher season stats: {len(result)} pitchers -> {cache_path.name}")
        return result

    # ── Batter season stats (bulk, 1 call per season) ─────────────────────

    def get_batter_season_stats(
        self,
        season: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return one row per batter for `season` with k_pct, bb_pct.
        (Plate-discipline o_swing/contact/swstr come from the Statcast snapshot.)

        Columns: batter (mlbam id), season, n_pa, k_pct, bb_pct
        """
        cache_path = self.cache_dir / f"statsapi_batter_season_{season}.parquet"
        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached StatsAPI batter season: {cache_path.name}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching StatsAPI batter season stats {season} (bulk)")
        try:
            data = self._get("stats", {
                "stats": "season", "group": "hitting", "season": season,
                "playerPool": "all", "limit": 4000, "gameType": "R",
            })
        except Exception as exc:
            logger.warning(f"StatsAPI batter season fetch failed for {season}: {exc}")
            return pd.DataFrame()

        rows = []
        for split in _splits(data):
            pid = split.get("player", {}).get("id")
            st = split.get("stat", {})
            if pid is None:
                continue
            pa = st.get("plateAppearances", 0) or 0
            k  = st.get("strikeOuts", 0) or 0
            bb = st.get("baseOnBalls", 0) or 0
            rows.append({"batter": int(pid), "season": season,
                         "n_pa": int(pa), "k": int(k), "bb": int(bb)})

        if not rows:
            logger.warning(f"No batter season rows returned for {season}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        pa = df["n_pa"].clip(lower=1)
        df["k_pct"]  = df["k"] / pa
        df["bb_pct"] = df["bb"] / pa
        df = df[df["n_pa"] >= 50].copy()

        keep = ["batter", "season", "k_pct", "bb_pct", "n_pa"]
        result = df[keep].reset_index(drop=True)
        result.to_parquet(cache_path, index=False)
        logger.info(f"Batter season stats: {len(result)} batters -> {cache_path.name}")
        return result

    # ── Pitcher game log (per-start, used for rolling + workload) ──────────

    def get_pitcher_gamelog(
        self,
        pitcher_id: int,
        seasons: Iterable[int],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return one row per start for a pitcher across `seasons`, ordered by date.

        Columns: pitcher, game_pk, game_date, strikeouts, pitches,
                 innings_pitched, batters_faced, is_home, opponent_team_id
        These feed k_rolling_{3,5,10,20}, days_rest, pitches_last, and the
        strikeouts target — fully replacing per-pitcher Baseball Savant scraping.
        """
        seasons = sorted(set(int(s) for s in seasons))
        tag = f"{seasons[0]}_{seasons[-1]}" if seasons else "none"
        cache_path = self.cache_dir / f"statsapi_gamelog_{pitcher_id}_{tag}.parquet"
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        all_rows: List[dict] = []
        for season in seasons:
            try:
                data = self._get(f"people/{pitcher_id}/stats", {
                    "stats": "gameLog", "group": "pitching",
                    "season": season, "gameType": "R",
                })
            except Exception as exc:
                logger.warning(f"gameLog fetch failed pid={pitcher_id} {season}: {exc}")
                continue

            for split in _splits(data):
                st = split.get("stat", {})
                gm = split.get("game", {})
                opp = split.get("opponent", {})
                all_rows.append({
                    "pitcher":          int(pitcher_id),
                    "game_pk":          gm.get("gamePk"),
                    "game_date":        split.get("date"),
                    "strikeouts":       int(st.get("strikeOuts", 0) or 0),
                    "pitches":          int(st.get("numberOfPitches", 0) or 0),
                    "innings_pitched":  _parse_ip(st.get("inningsPitched")),
                    "batters_faced":    int(st.get("battersFaced", 0) or 0),
                    "is_home":          bool(split.get("isHome", False)),
                    "opponent_team_id": opp.get("id"),
                })

        if not all_rows:
            logger.warning(f"No gameLog rows for pid={pitcher_id}")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        df.to_parquet(cache_path, index=False)
        return df


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    f = StatsAPIFetcher()

    pit = f.get_pitcher_season_stats(2024, force_refresh=True)
    print(f"\nPitcher season: {len(pit)} rows")
    print(pit.head().to_string(index=False))

    bat = f.get_batter_season_stats(2024, force_refresh=True)
    print(f"\nBatter season: {len(bat)} rows")
    print(bat.head().to_string(index=False))

    log = f.get_pitcher_gamelog(594798, [2024], force_refresh=True)
    print(f"\nGame log (deGrom 2024): {len(log)} starts")
    print(log.to_string(index=False))
