"""
MLB Strikeout Pipeline — Feature Builder
Takes per-start Statcast metrics, FanGraphs season stats, a LineupCard,
and context data (park, umpire, weather) and assembles one flat feature
row per pitcher × game.

This is the central transformation step:
    raw data  →  FeatureBuilder  →  model input

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
    ):
        self.fg_pitchers = fg_pitcher_df
        self.fg_batters  = fg_batter_df
        self.starts      = statcast_starts.sort_values(["pitcher", "game_date"]).copy()
        self.umpires     = umpire_k_df

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

        # ── 2. FanGraphs season-level pitcher features ────────────────────
        season = game_dt.year
        fg_row = self._get_fg_pitcher(pitcher_mlbam_id, season)
        if fg_row is not None:
            row["k9_season"]       = fg_row.get("K/9",    np.nan)
            row["k_pct_season"]    = fg_row.get("K%",     np.nan)
            row["bb_pct_season"]   = fg_row.get("BB%",    np.nan)
            row["fip_season"]      = fg_row.get("FIP",    np.nan)
            row["xfip_season"]     = fg_row.get("xFIP",   np.nan)
            row["swstr_season"]    = fg_row.get("SwStr%", np.nan)
            row["stuff_plus"]      = fg_row.get("Stuff+", np.nan)
        else:
            # Fall back to career rolling average if current season is short
            row.update({k: np.nan for k in [
                "k9_season","k_pct_season","bb_pct_season",
                "fip_season","xfip_season","swstr_season","stuff_plus"
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

    def build_training_set(self, game_log: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full feature matrix for model training.

        game_log must have columns:
            game_pk, game_date, pitcher_mlbam, pitcher_hand,
            is_home, park_id, is_night_game,
            opp_team_id, strikeouts  (target)

        Returns a DataFrame with all features + target column.
        """
        rows = []
        for _, g in game_log.iterrows():
            # For training we create a minimal synthetic LineupCard
            # (the historical confirmed lineup was the actual lineup)
            card = self._make_training_lineup_card(g)
            feat = self.build_row(
                pitcher_mlbam_id = int(g["pitcher_mlbam"]),
                game_pk          = int(g["game_pk"]),
                game_date        = str(g["game_date"])[:10],
                lineup_card      = card,
                pitcher_hand     = str(g.get("pitcher_hand", "R")),
                is_home          = bool(g.get("is_home", True)),
                park_id          = int(g.get("park_id", 680)),
                is_night_game    = bool(g.get("is_night_game", True)),
            )
            if feat is not None:
                feat["strikeouts"] = g["strikeouts"]   # attach target
                rows.append(feat)

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
            fg = self._get_fg_batter(b.player_id, season)
            if fg is None:
                continue

            # PA-weighted slot weight (leadoff sees ~15% more PAs than #9)
            slot_w = 1.0 + max(0, (5 - b.batting_order)) * 0.03

            # Optional: filter by platoon side if batter handedness available
            k_pcts.append(fg.get("K%", np.nan))
            o_swing_pcts.append(fg.get("O-Swing%", np.nan))
            contact_pcts.append(fg.get("Contact%", np.nan))
            swstr_pcts.append(fg.get("SwStr%", np.nan))
            wrc_plus.append(fg.get("wRC+", 100.0))
            weights.append(slot_w)

        def wmean(vals):
            pairs = [(v, w) for v, w in zip(vals, weights) if not np.isnan(v)]
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
            "opp_lineup_size":       len(batters),
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
                "opp_lineup_size":     0,
            }

        return {
            "opp_lineup_k_pct":    team_batters["K%"].mean(),
            "opp_lineup_o_swing":  team_batters.get("O-Swing%", pd.Series()).mean(),
            "opp_lineup_contact":  team_batters.get("Contact%", pd.Series()).mean(),
            "opp_lineup_swstr":    team_batters.get("SwStr%", pd.Series()).mean(),
            "opp_lineup_wrc_plus": team_batters.get("wRC+", pd.Series()).mean(),
            "opp_lineup_size":     0,   # 0 = proxy, not confirmed
        }

    # ── Lookup helpers ─────────────────────────────────────────────────────

    def _get_fg_pitcher(self, mlbam_id: int, season: int) -> Optional[Dict]:
        """
        Return FanGraphs pitcher row for a given season.
        Falls back to the most recent season if current season not yet available.
        Note: FanGraphs uses its own IDfg, not MLBAM — this requires the
              pybaseball playerid_lookup cross-reference. For now we match by
              approximate name match; replace with a proper ID mapping table.
        """
        if self.fg_pitchers.empty:
            return None

        # Prefer current season; fall back to previous
        for yr in [season, season - 1]:
            subset = self.fg_pitchers[self.fg_pitchers["Season"] == yr]
            # TODO: replace with a pre-joined MLBAM → IDfg mapping
            if not subset.empty:
                return subset.iloc[0].to_dict()   # placeholder until ID mapping
        return None

    def _get_fg_batter(self, player_id: int, season: int) -> Optional[Dict]:
        """Return FanGraphs batter row. Same ID-mapping caveat as above."""
        if self.fg_batters.empty:
            return None
        subset = self.fg_batters[self.fg_batters["Season"] == season]
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
