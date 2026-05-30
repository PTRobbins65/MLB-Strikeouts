"""
MLB Strikeout Pipeline — Feature Builder
Takes per-start Statcast metrics, FanGraphs season stats, a LineupCard,
and context data (park, umpire, weather) and assembles one flat feature
row per pitcher × game.

This is the central transformation step:
    raw data  ->  FeatureBuilder  ->  model input

Usage
-----
    from feature_builder import FeatureBuilder
    builder = FeatureBuilder(fg_pitcher_df, fg_batter_df, statcast_starts_df)
    row = builder.build_row(
        pitcher_mlbam_id = 477132,
        game_pk          = 717465,
        game_date        = "2024-06-15",
        lineup_card      = confirmed_lineup_card,
        pitcher_hand     = "L",
        is_home          = True,
        park_id          = 22,
    )
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import ROLLING_WINDOWS, BASE_FEATURE_GROUPS
from lineup_manager import BatterSlot, LineupCard

logger = logging.getLogger(__name__)


def _safe_int(val, default=0) -> int:
    """Convert val to int, returning default for None / NaN / non-numeric."""
    try:
        return int(val) if val == val else default  # val != val catches float NaN
    except (TypeError, ValueError):
        return default


# Lazy import: IdMapper may not be built on first run
try:
    from id_mapper import IdMapper as _IdMapper
except ImportError:
    _IdMapper = None


# ── Park strikeout factors ──────────────────────────────────────────────────
# Source: multi-year FanGraphs park factors for K.
# Key = MLB venue ID, value = K park factor (1.00 = neutral).
# Update annually; this table covers recent seasons as a reasonable default.
PARK_K_FACTORS: Dict[int, float] = {
    1:  1.02,   # Oriole Park at Camden Yards
    2:  0.99,   # Fenway Park
    3:  1.00,   # Yankee Stadium
    4:  0.98,   # Rogers Centre
    5:  0.97,   # Guaranteed Rate Field
    7:  1.00,   # Progressive Field
    9:  1.01,   # Comerica Park
    10: 0.98,   # Kauffman Stadium
    11: 0.99,   # American Family Field
    12: 1.00,   # Target Field
    14: 1.03,   # Minute Maid Park
    15: 0.99,   # Angels Stadium
    16: 1.01,   # Oakland Coliseum
    17: 1.02,   # T-Mobile Park
    18: 1.00,   # Globe Life Field
    19: 1.01,   # Tropicana Field
    20: 0.99,   # SunTrust (Truist) Park
    21: 1.01,   # Wrigley Field
    22: 0.97,   # Great American Ball Park
    23: 1.02,   # Coors Field  ← surprisingly normal K rates
    24: 0.99,   # Chase Field
    25: 1.00,   # Dodger Stadium
    26: 1.01,   # Petco Park
    27: 1.01,   # Oracle Park
    28: 1.03,   # loanDepot Park
    29: 1.01,   # Citi Field
    30: 1.00,   # Citizens Bank Park
    31: 1.01,   # PNC Park
    32: 0.99,   # Busch Stadium
    33: 0.99,   # American Family (MIL)
    34: 1.02,   # Nationals Park
    680: 1.00,  # default / unknown
}


class FeatureBuilder:
    """
    Assembles model-ready feature rows from multiple data sources.

    Parameters
    ----------
    fg_pitcher_df    : FanGraphs pitcher stats (all seasons, from HistoricalDataFetcher)
    fg_batter_df     : FanGraphs batter stats  (all seasons)
    statcast_starts  : per-start Statcast metrics (from compute_per_start_metrics)
    umpire_k_df      : optional DataFrame with umpire_id, k_pct columns
    """

    def __init__(
        self,
        fg_pitcher_df: pd.DataFrame,
        fg_batter_df: pd.DataFrame,
        statcast_starts: pd.DataFrame,
        umpire_k_df: Optional[pd.DataFrame] = None,
        statcast_season: Optional[pd.DataFrame] = None,
        sc_batter_stats: Optional[pd.DataFrame] = None,
    ):
        self.fg_pitchers = fg_pitcher_df
        self.fg_batters  = fg_batter_df
        if not statcast_starts.empty:
            self.starts = statcast_starts.sort_values(["pitcher", "game_date"]).copy()
        else:
            self.starts = pd.DataFrame(columns=["pitcher", "game_date", "strikeouts",
                                                 "whiff_pct", "csw_pct", "zone_pct",
                                                 "o_swing_pct", "avg_velo", "pitches"])
        self.umpires     = umpire_k_df

        # Statcast-derived pitcher season stats (replaces FanGraphs when available).
        # Index by (pitcher MLBAM, season year) for O(1) lookup.
        if statcast_season is not None and not statcast_season.empty:
            self._sc_season_idx = statcast_season.set_index(["pitcher", "season"])
            logger.info(
                f"Statcast season stats loaded: "
                f"{len(statcast_season)} pitcher-seasons available"
            )
        else:
            self._sc_season_idx = None

        # Statcast-derived batter season stats (replaces FanGraphs batter data).
        # Index by (batter MLBAM, season year) for O(1) lookup.
        if sc_batter_stats is not None and not sc_batter_stats.empty:
            self._sc_batter_idx = sc_batter_stats.set_index(["batter", "season"])
            logger.info(
                f"Statcast batter stats loaded: "
                f"{len(sc_batter_stats):,} batter-seasons available"
            )
        else:
            self._sc_batter_idx = None

        # ID mapper for MLBAM -> FanGraphs IDfg lookups (fallback only)
        self._id_mapper = None
        if _IdMapper is not None:
            try:
                self._id_mapper = _IdMapper()
            except Exception as exc:
                logger.warning(f"IdMapper unavailable — FanGraphs lookups will return None: {exc}")

    # ── Main entry point ───────────────────────────────────────────────────

    def build_row(
        self,
        pitcher_mlbam_id: int,
        game_pk: int,
        game_date: str,
        lineup_card: LineupCard,
        pitcher_hand: str = "R",
        is_home: bool = True,
        park_id: int = 680,
        is_night_game: bool = True,
        umpire_id: Optional[int] = None,
    ) -> Dict:
        """
        Return a single feature dict ready for model inference or training.
        Returns None if critical features are unavailable.
        """
        game_dt = pd.Timestamp(game_date)

        row: Dict = {
            "game_pk":         game_pk,
            "game_date":       game_date,
            "pitcher_mlbam":   pitcher_mlbam_id,
            "pitcher_hand":    pitcher_hand,
            "is_home":         int(is_home),
            "is_night_game":   int(is_night_game),
            "park_k_factor":   PARK_K_FACTORS.get(park_id, 1.00),
            "lineup_confirmed": int(lineup_card.confirmed),
        }

        # ── 1. Pitcher rolling Statcast features ──────────────────────────
        pitcher_starts = self.starts[
            (self.starts["pitcher"] == pitcher_mlbam_id) &
            (self.starts["game_date"] < game_dt)
        ].tail(max(ROLLING_WINDOWS))

        if pitcher_starts.empty:
            logger.warning(f"No prior starts found for mlbam_id={pitcher_mlbam_id}")
            return None

        for w in ROLLING_WINDOWS:
            recent = pitcher_starts.tail(w)
            suffix = f"_{w}"
            row[f"k_rolling{suffix}"]       = recent["strikeouts"].mean()
            row[f"whiff_pct{suffix}"]        = recent["whiff_pct"].mean()
            row[f"csw_pct{suffix}"]          = recent["csw_pct"].mean()
            row[f"zone_pct{suffix}"]         = recent["zone_pct"].mean()
            row[f"o_swing_pct{suffix}"]      = recent["o_swing_pct"].mean()
            row[f"avg_velo{suffix}"]         = recent["avg_velo"].mean()

        # Most-recent start workload
        last_start = pitcher_starts.iloc[-1]
        row["days_rest"]        = (game_dt - last_start["game_date"]).days
        row["pitches_last"]     = last_start.get("pitches", np.nan)

        # ── 2. Season-level pitcher features ──────────────────────────────
        # Priority: Statcast-derived stats (computed from our own cache, no
        # FanGraphs dependency) → FanGraphs (if unblocked) → NaN fallback.
        season = game_dt.year
        sc_row = self._get_statcast_season(pitcher_mlbam_id, season)
        if sc_row is not None:
            row["k9_season"]    = sc_row.get("k9_season",    np.nan)
            row["k_pct_season"] = sc_row.get("k_pct_season", np.nan)
            row["bb_pct_season"]= sc_row.get("bb_pct_season",np.nan)
            row["fip_season"]   = sc_row.get("fip_season",   np.nan)
            row["swstr_season"] = sc_row.get("swstr_season", np.nan)
            # xFIP and Stuff+ are FanGraphs-only — leave NaN
            row["xfip_season"]  = np.nan
            row["stuff_plus"]   = np.nan
            logger.debug(
                f"Statcast season stats used for mlbam={pitcher_mlbam_id} "
                f"season={season}: k_pct={row['k_pct_season']:.3f}, "
                f"swstr={row['swstr_season']:.3f}, fip={row['fip_season']:.2f}"
            )
        else:
            # Fallback: try FanGraphs (may be blocked)
            fg_row = self._get_fg_pitcher(pitcher_mlbam_id, season)
            if fg_row is not None:
                row["k9_season"]    = fg_row.get("K/9",    np.nan)
                row["k_pct_season"] = fg_row.get("K%",     np.nan)
                row["bb_pct_season"]= fg_row.get("BB%",    np.nan)
                row["fip_season"]   = fg_row.get("FIP",    np.nan)
                row["xfip_season"]  = fg_row.get("xFIP",   np.nan)
                row["swstr_season"] = fg_row.get("SwStr%", np.nan)
                row["stuff_plus"]   = fg_row.get("Stuff+", np.nan)
            else:
                row.update({k: np.nan for k in [
                    "k9_season", "k_pct_season", "bb_pct_season",
                    "fip_season", "xfip_season", "swstr_season", "stuff_plus",
                ]})

        # ── 3. Opponent lineup features ───────────────────────────────────
        opp_batters = lineup_card.away_batters if is_home else lineup_card.home_batters

        if opp_batters:
            row.update(self._build_lineup_features(opp_batters, pitcher_hand, season))
        else:
            # Lineup not yet confirmed — use team-level projections
            opp_team_id = lineup_card.away_team_id if is_home else lineup_card.home_team_id
            row.update(self._build_team_level_features(opp_team_id, pitcher_hand, season))
            logger.debug(f"Using team-level lineup proxy for game_pk={game_pk}")

        # ── 4. Umpire feature ─────────────────────────────────────────────
        if self.umpires is not None and umpire_id is not None:
            u_row = self.umpires[self.umpires["umpire_id"] == umpire_id]
            row["umpire_k_pct"] = u_row["k_pct"].values[0] if not u_row.empty else np.nan
        else:
            row["umpire_k_pct"] = np.nan

        return row

    # ── Batch builder (training set) ───────────────────────────────────────

    def build_training_set(
        self,
        game_log: pd.DataFrame,
        lineup_lookup: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Build the full feature matrix for model training.

        game_log must have columns:
            game_pk, game_date, pitcher_mlbam, pitcher_hand,
            is_home, park_id, is_night_game,
            home_team, away_team, home_team_id, away_team_id,
            strikeouts  (target)

        lineup_lookup : optional dict of game_pk -> LineupCard built from
                        historical Statcast data. When provided, real opponent
                        batters replace the empty synthetic LineupCards,
                        populating opp_lineup_* features for every training row.
        """
        hits   = 0
        misses = 0
        rows   = []

        for _, g in game_log.iterrows():
            game_pk = int(g["game_pk"])

            if lineup_lookup and game_pk in lineup_lookup:
                card = lineup_lookup[game_pk]
                hits += 1
            else:
                card = self._make_training_lineup_card(g)
                misses += 1

            feat = self.build_row(
                pitcher_mlbam_id = int(g["pitcher_mlbam"]),
                game_pk          = game_pk,
                game_date        = str(g["game_date"])[:10],
                lineup_card      = card,
                pitcher_hand     = str(g.get("pitcher_hand", "R")),
                is_home          = bool(g.get("is_home", True)),
                park_id          = _safe_int(g.get("park_id"), default=680),
                is_night_game    = bool(g.get("is_night_game", True)),
            )
            if feat is not None:
                feat["strikeouts"] = g["strikeouts"]
                rows.append(feat)

        if lineup_lookup:
            logger.info(
                f"Lineup lookup coverage: {hits:,} hits / {misses:,} misses "
                f"({100 * hits / max(hits + misses, 1):.1f}% of training rows have real lineups)"
            )

        df = pd.DataFrame(rows)
        logger.info(f"Training set: {len(df):,} rows, {len(df.columns)} features")
        return df

    # ── Feature sub-builders ───────────────────────────────────────────────

    def _build_lineup_features(
        self,
        batters: List[BatterSlot],
        pitcher_hand: str,
        season: int,
    ) -> Dict:
        """
        Aggregate batter K-profile features across the confirmed/projected lineup.
        Weights top-of-order batters slightly more (they see more PAs).
        """
        k_pcts      = []
        o_swing_pcts= []
        contact_pcts= []
        swstr_pcts  = []
        wrc_plus    = []
        weights     = []

        for b in batters:
            # PA-weighted slot weight (leadoff sees ~15% more PAs than #9)
            slot_w = 1.0 + max(0, (5 - b.batting_order)) * 0.03

            # Try Statcast-derived batter stats first (always available;
            # no FanGraphs dependency).  Fall back to FanGraphs if needed.
            sc = self._get_statcast_batter(b.player_id, season)
            if sc is not None:
                k_pcts.append(sc.get("k_pct",       np.nan))
                o_swing_pcts.append(sc.get("o_swing_pct", np.nan))
                contact_pcts.append(sc.get("contact_pct", np.nan))
                swstr_pcts.append(sc.get("swstr_pct",   np.nan))
                # wRC+ cannot be derived from raw Statcast without park adjustments
                wrc_plus.append(np.nan)
                weights.append(slot_w)
                continue

            # Fallback: FanGraphs (may be blocked)
            fg = self._get_fg_batter(b.player_id, season)
            if fg is None:
                continue

            k_pcts.append(fg.get("K%",        np.nan))
            o_swing_pcts.append(fg.get("O-Swing%",  np.nan))
            contact_pcts.append(fg.get("Contact%",  np.nan))
            swstr_pcts.append(fg.get("SwStr%",    np.nan))
            wrc_plus.append(fg.get("wRC+", 100.0))
            weights.append(slot_w)

        def wmean(vals):
            # pd.notna tolerates None / NaN / object-dtype missing values that
            # arise when StatsAPI counting stats are merged with snapshot stuff
            # (np.isnan would raise on a Python None). Coerce survivors to float.
            pairs = []
            for v, w in zip(vals, weights):
                if pd.notna(v):
                    try:
                        pairs.append((float(v), w))
                    except (TypeError, ValueError):
                        continue
            if not pairs:
                return np.nan
            vs, ws = zip(*pairs)
            return np.average(vs, weights=ws)

        return {
            "opp_lineup_k_pct":      wmean(k_pcts),
            "opp_lineup_o_swing":    wmean(o_swing_pcts),
            "opp_lineup_contact":    wmean(contact_pcts),
            "opp_lineup_swstr":      wmean(swstr_pcts),
            "opp_lineup_wrc_plus":   wmean(wrc_plus),
            # MLB always starts 9 batters — hard-code 9 so the model never
            # sees a value outside its training distribution of 8–9.
            "opp_lineup_size":       9,
        }

    def _build_team_level_features(
        self,
        team_id: int,
        pitcher_hand: str,
        season: int,
    ) -> Dict:
        """
        When the confirmed lineup isn't available yet, fall back to team-level
        average K% from FanGraphs batting stats for that season.
        This gives a rougher but still useful signal.
        """
        team_batters = self.fg_batters[
            (self.fg_batters["Season"] == season) &
            (self.fg_batters.get("teamId", pd.Series()) == team_id)
        ] if "teamId" in self.fg_batters.columns else pd.DataFrame()

        if team_batters.empty:
            return {
                "opp_lineup_k_pct":    np.nan,
                "opp_lineup_o_swing":  np.nan,
                "opp_lineup_contact":  np.nan,
                "opp_lineup_swstr":    np.nan,
                "opp_lineup_wrc_plus": np.nan,
                "opp_lineup_size":     9,   # always 9 — MLB never starts fewer
            }

        return {
            "opp_lineup_k_pct":    team_batters["K%"].mean(),
            "opp_lineup_o_swing":  team_batters.get("O-Swing%", pd.Series()).mean(),
            "opp_lineup_contact":  team_batters.get("Contact%", pd.Series()).mean(),
            "opp_lineup_swstr":    team_batters.get("SwStr%", pd.Series()).mean(),
            "opp_lineup_wrc_plus": team_batters.get("wRC+", pd.Series()).mean(),
            "opp_lineup_size":     9,   # always 9
        }

    # ── Lookup helpers ─────────────────────────────────────────────────────

    def _get_statcast_season(self, mlbam_id: int, season: int) -> Optional[Dict]:
        """
        Return Statcast-derived season stats dict for a pitcher.

        Tries current season first, falls back to previous season if the
        pitcher hasn't yet accumulated 50 PA in the current year (e.g. early
        April when season stats are sparse).
        """
        if self._sc_season_idx is None:
            return None

        for yr in [season, season - 1]:
            try:
                row = self._sc_season_idx.loc[(mlbam_id, yr)]
                # loc can return a Series (single match) or DataFrame (multiple)
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return row.to_dict()
            except KeyError:
                continue
        return None

    def _get_statcast_batter(self, batter_mlbam: int, season: int) -> Optional[Dict]:
        """
        Return Statcast-derived batter season stats dict.
        Tries current season first, falls back one year (early-season sparse data).
        """
        if self._sc_batter_idx is None:
            return None
        for yr in [season, season - 1]:
            try:
                row = self._sc_batter_idx.loc[(batter_mlbam, yr)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return row.to_dict()
            except KeyError:
                continue
        return None

    def _get_fg_pitcher(self, mlbam_id: int, season: int) -> Optional[Dict]:
        """
        Return FanGraphs pitcher row for a given season.
        Uses MLBAM -> IDfg mapping via IdMapper (Chadwick Bureau register).
        Falls back to previous season if current season isn't yet available.
        """
        if self.fg_pitchers.empty or self._id_mapper is None:
            return None

        fg_id = self._id_mapper.mlbam_to_fg(mlbam_id)
        if fg_id is None:
            return None

        for yr in [season, season - 1]:
            subset = self.fg_pitchers[
                (self.fg_pitchers["Season"] == yr) &
                (self.fg_pitchers["IDfg"] == fg_id)
            ]
            if not subset.empty:
                return subset.iloc[0].to_dict()
        return None

    def _get_fg_batter(self, player_id: int, season: int) -> Optional[Dict]:
        """Return FanGraphs batter row matched by MLBAM -> IDfg mapping."""
        if self.fg_batters.empty or self._id_mapper is None:
            return None

        fg_id = self._id_mapper.mlbam_to_fg(player_id)
        if fg_id is None:
            return None

        subset = self.fg_batters[
            (self.fg_batters["Season"] == season) &
            (self.fg_batters["IDfg"] == fg_id)
        ]
        if not subset.empty:
            return subset.iloc[0].to_dict()
        return None

    # ── Training helper ────────────────────────────────────────────────────

    @staticmethod
    def _make_training_lineup_card(game_row: pd.Series) -> LineupCard:
        """Create a minimal LineupCard for training (no real batters needed here)."""
        return LineupCard(
            game_pk      = int(game_row.get("game_pk", 0)),
            game_date    = str(game_row.get("game_date", ""))[:10],
            home_team    = str(game_row.get("home_team", "")),
            away_team    = str(game_row.get("away_team", "")),
            home_team_id = int(game_row.get("home_team_id", 0)),
            away_team_id = int(game_row.get("away_team_id", 0)),
            confirmed    = True,
        )
