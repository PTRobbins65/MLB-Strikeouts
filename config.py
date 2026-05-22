"""
MLB Strikeout Prediction Pipeline — Configuration
All constants, paths, and data source settings live here.
"""

from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "logs"

# Raw data cache directories
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
FEATURES_DIR   = DATA_DIR / "features"

for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── MLB StatsAPI Settings ─────────────────────────────────────────────────────
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Hydration options for the schedule endpoint
SCHEDULE_HYDRATE = (
    "probablePitcher(note),"
    "lineups,"
    "team,"
    "linescore,"
    "game(content(summary))"
)

# Game status codes that mean lineup is confirmed
CONFIRMED_STATUS_CODES = {
    "Pre-Game",   # ~30 min before first pitch — lineup card submitted
    "Warmup",
    "In Progress",
    "Manager challenge",
    "Delayed",
    "Suspended",
    "Final",
    "Game Over",
    "Completed Early",
}

# ── Statcast / pybaseball Settings ────────────────────────────────────────────
# Rolling windows (games) used in feature engineering
ROLLING_WINDOWS = [3, 5, 10, 20]   # last-N-starts windows
SEASON_START_YEAR = 2015            # first year Statcast data is reliable

# Key Statcast columns we pull for pitchers
STATCAST_PITCHER_COLS = [
    "game_date", "pitcher", "player_name",
    "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z",                        # horizontal / vertical movement
    "release_pos_x", "release_pos_z",         # release point
    "release_extension",
    "plate_x", "plate_z",                     # location at plate
    "pitch_type", "pitch_name",
    "description",                            # swing, whiff, called_strike, etc.
    "zone",
    "type",                                   # B / S / X
    "bb_type",
    "events",                                 # strikeout, walk, etc.
    "strikes", "balls",
    "outs_when_up",
    "inning",
    "stand",                                  # batter handedness
    "p_throws",                               # pitcher hand
]

# Descriptions that count as a whiff (swinging strike)
WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "foul_tip",
}

# Descriptions that count toward CSW (called strike + whiff)
CSW_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {"called_strike"}

# ── FanGraphs Settings ────────────────────────────────────────────────────────
# pybaseball pitching_stats columns we want to retain
FANGRAPHS_PITCHER_COLS = [
    "IDfg", "Name", "Season", "Team",
    "G", "GS", "IP",
    "K/9", "BB/9", "K%", "BB%", "K-BB%",
    "AVG", "WHIP", "FIP", "xFIP", "ERA",
    "SwStr%", "CSW%",
    "O-Swing%", "Z-Swing%", "Swing%",
    "O-Contact%", "Z-Contact%", "Contact%",
    "Zone%",
    "Stuff+", "Location+", "Pitching+",      # PitcherList modern metrics
]

# ── Lineup / Projection Settings ─────────────────────────────────────────────
# How often (seconds) to re-poll the StatsAPI for lineup updates (pre-game)
LINEUP_POLL_INTERVAL_SECONDS = 300   # 5 minutes

# Hours before first pitch at which we build the "projected" lineup
PROJECTED_LINEUP_HOURS_BEFORE = 24

# ── Batter Feature Cols (for opponent lineup aggregation) ─────────────────────
BATTER_FEATURES = [
    "K%",          # season strikeout rate
    "BB%",
    "O-Swing%",    # chase rate
    "Contact%",    # overall contact
    "SwStr%",      # swinging strike rate
    "wRC+",        # overall offensive value (context for lineup quality)
    "PA",          # playing time signal
]

# ── Model Settings ────────────────────────────────────────────────────────────
TARGET_COL     = "strikeouts"       # what we're predicting
MODEL_DIR      = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Features used in the final model (will be expanded after EDA)
BASE_FEATURE_GROUPS = {
    "pitcher_stuff":    ["release_speed_mean", "release_spin_rate_mean",
                         "pfx_x_mean", "pfx_z_mean", "release_extension_mean"],
    "command_results":  ["whiff_pct", "csw_pct", "zone_pct", "o_swing_pct",
                         "k9_rolling5", "k_pct_rolling5", "fip_season"],
    "workload":         ["days_rest", "pitches_last_start", "ip_last_start",
                         "k_rolling3", "k_rolling10"],
    "opponent_lineup":  ["opp_k_pct_vs_hand", "opp_lineup_avg_k_pct",
                         "opp_lineup_avg_o_swing", "lineup_confirmed"],
    "context":          ["park_k_factor", "is_home", "is_night_game",
                         "umpire_k_pct"],
}
