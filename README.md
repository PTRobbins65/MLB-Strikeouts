# MLB Pitcher Strikeout Prediction — Data Pipeline

A modular Python pipeline that ingests historical MLB data, builds
projected and confirmed opponent lineups, and engineers features for
a daily pitcher strikeout prediction model.

---

## Architecture

```
mlb_strikeout_pipeline/
├── config.py           — constants, paths, column lists
├── data_fetcher.py     — Statcast + FanGraphs data, disk caching
├── lineup_manager.py   — projected & confirmed lineup lifecycle
├── feature_builder.py  — feature engineering (pitcher + lineup + context)
├── pipeline.py         — daily orchestrator / CLI
├── requirements.txt
└── data/
    ├── raw/            — cached Parquet files (Statcast, FanGraphs)
    ├── processed/      — intermediate cleaned frames
    └── features/       — final feature matrices per date
```

---

## Data Sources

| Source | Library | What we pull |
|---|---|---|
| Baseball Savant (Statcast) | `pybaseball` | Pitch-level data (velo, spin, movement, outcome) 2015→ |
| FanGraphs | `pybaseball` | Pitcher K%, SwStr%, CSW%, Stuff+, FIP per season |
| FanGraphs batting | `pybaseball` | Batter K%, O-Swing%, Contact% per season |
| MLB StatsAPI | `MLB-StatsAPI` | Schedule, probable pitchers, confirmed lineups |

---

## Lineup Lifecycle

```
T-24h           T-4h to T-2h         T-0 (first pitch)
────────────────────────────────────────────────────────
  PROJECTED          CONFIRMED              GAME
  (history-          (StatsAPI             LOCKED
   frequency)         publishes)
       │                  │
       └──── polling ──────┘
             every 5 min
```

**Phase 1 — Projected**
Built from the team's last 15 confirmed lineups.  For each batting slot
(1–9) we pick the player with the most appearances in that slot.  This
gives a reasonable pre-game estimate that's ready 24 hours in advance.

**Phase 2 — Confirmed**
A background polling thread checks `statsapi.mlb.com/api/v1/game/{pk}/boxscore`
every 5 minutes.  When ≥ 7 batters appear in the batting order the lineup
is marked confirmed and the projected batters are replaced with the real ones.

Teams typically release lineups **2–4 hours before first pitch**, with
most managers posting between 11 AM and 1 PM ET for evening games.

---

## Feature Groups

### Pitcher stuff (Statcast rolling)
| Feature | Window | Description |
|---|---|---|
| `k_rolling_N` | 3/5/10/20 starts | Average strikeouts per start |
| `whiff_pct_N` | 3/5/10/20 starts | Swinging strike / swing |
| `csw_pct_N` | 3/5/10/20 starts | Called strike + whiff / pitch |
| `zone_pct_N` | 3/5/10/20 starts | In-zone pitch rate |
| `o_swing_pct_N` | 3/5/10/20 starts | Out-of-zone swing rate |
| `avg_velo_N` | 3/5/10/20 starts | Mean fastball velocity |

### Pitcher season (FanGraphs)
`k9_season`, `k_pct_season`, `fip_season`, `xfip_season`, `swstr_season`,
`stuff_plus`, `bb_pct_season`

### Opponent lineup
| Feature | Notes |
|---|---|
| `opp_lineup_k_pct` | PA-weighted avg K% of opposing batters |
| `opp_lineup_o_swing` | Chase rate — how often they swing at balls |
| `opp_lineup_contact` | Contact% — inverse of swing-and-miss ability |
| `opp_lineup_swstr` | Swinging strike rate |
| `opp_lineup_wrc_plus` | Overall lineup quality signal |
| `opp_lineup_size` | 9 = confirmed, 0 = team-level proxy |
| `lineup_confirmed` | 1 = real lineup; 0 = projected |

### Workload & context
`days_rest`, `pitches_last`, `park_k_factor`, `is_home`,
`is_night_game`, `umpire_k_pct`

---

## Quickstart

```bash
pip install -r requirements.txt

# Run today's pipeline (builds projections immediately, polls for confirmations)
python pipeline.py --show

# Run for a specific past date (no polling needed)
python pipeline.py --date 2024-06-15 --no-wait --show

# See all options
python pipeline.py --help
```

---

## Extending

**Adding umpire K% data**
Pass a DataFrame with columns `umpire_id, k_pct` to `FeatureBuilder`.
Umpire data can be scraped from Baseball Reference's umpire pages.

**Adding weather features**
Pull wind speed, temperature, and humidity from a weather API
(OpenWeather works well) using the venue coordinates from the StatsAPI
`venues` endpoint.  Add columns to the feature row in `pipeline.py`.

**Training the model**
`feature_builder.build_training_set(game_log_df)` builds the full
historical feature matrix.  Pass the result to the XGBoost / Poisson
model training script (next pipeline phase).

---

## ID Mapping Note
FanGraphs uses its own `IDfg` system, not MLBAM IDs. The current
`_get_fg_pitcher` / `_get_fg_batter` helpers in `feature_builder.py`
use placeholder logic.  Before training, build a cross-reference table:

```python
from pybaseball import playerid_lookup
# e.g. playerid_lookup("degrom", "jacob") returns mlbam, bbref, IDfg, retro IDs
```

Store this as `data/player_id_map.parquet` and join in `FeatureBuilder.__init__`.
