"""
MLB Strikeout Prediction — FastAPI Backend

Architecture
------------
This API is a READ-ONLY cache layer. It serves from pre-computed parquet
files written by pipeline.py. It does NOT run the pipeline on request —
that takes several minutes. Instead:

  1. Run pipeline.py on a schedule (cron / Railway cron job) each morning.
  2. This API serves those results instantly all day.
  3. Call POST /admin/refresh to trigger a background pipeline re-run
     (e.g. after lineups are confirmed).

Endpoints
---------
  GET  /health                        — uptime + model version
  GET  /predictions/today             — today's predictions (JSON array)
  GET  /predictions?date=YYYY-MM-DD   — predictions for any date
  GET  /metrics?days=30               — rolling accuracy summary
  GET  /metrics/trend?days=90         — daily MAE time series for charting
  POST /admin/refresh?date=YYYY-MM-DD — trigger background pipeline run

Deploy
------
  uvicorn api:app --host 0.0.0.0 --port 8000

  # Or with auto-reload for local dev:
  uvicorn api:app --reload
"""

import logging
import os
import subprocess
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from accuracy_tracker import AccuracyTracker
from config import FEATURES_DIR, MODEL_DIR, features_key
from storage import get_storage

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("api")

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MLB Strikeout Prediction API",
    description="Daily pitcher strikeout projections for prop betting.",
    version="1.0.0",
)

# Allow all origins for now — restrict to your Lovable domain once deployed
# e.g. allow_origins=["https://your-app.lovable.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic response models ───────────────────────────────────────────────────


class PredictionRow(BaseModel):
    """One row in the predictions response — one pitcher per game."""
    pitcher_name:     str
    pitcher_mlbam:    int
    team:             str
    opponent:         str
    is_home:          bool
    predicted_k:      float
    lineup_confirmed: int           # 1 = confirmed, 0 = projected
    model_version:    str

    # Key context features (shown in UI cards)
    k_rolling_5:       Optional[float] = None
    k_rolling_3:       Optional[float] = None
    whiff_pct_5:       Optional[float] = None
    csw_pct_5:         Optional[float] = None
    opp_lineup_k_pct:  Optional[float] = None
    opp_lineup_size:   Optional[float] = None
    park_k_factor:     Optional[float] = None
    days_rest:         Optional[float] = None
    game_date:         Optional[str]   = None
    game_pk:           Optional[int]   = None


class MetricsSummary(BaseModel):
    n_predictions:  int
    n_graded:       int
    mae:            Optional[float]
    rmse:           Optional[float]
    bias:           Optional[float]
    within_1_k:     Optional[float]
    within_2_k:     Optional[float]
    mae_confirmed:  Optional[float]
    mae_projected:  Optional[float]
    window_days:    Optional[int]
    start_date:     str
    end_date:       str


class TrendRow(BaseModel):
    game_date:       str
    n_graded:        int
    daily_mae:       float
    rolling_mae_7d:  float
    bias:            float


class HealthResponse(BaseModel):
    status:          str
    model_version:   str
    model_type:      str
    predictions_today: int
    timestamp:       str


class RefreshResponse(BaseModel):
    status: str
    message: str
    date: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_model_info() -> dict:
    """Return model type and version (file modification date)."""
    xgb_path = MODEL_DIR / "strikeout_xgb.json"
    glm_path  = MODEL_DIR / "strikeout_glm.joblib"

    if xgb_path.exists():
        mtime   = datetime.fromtimestamp(xgb_path.stat().st_mtime)
        return {"type": "XGBoost", "version": mtime.strftime("%Y%m%d")}
    if glm_path.exists():
        mtime   = datetime.fromtimestamp(glm_path.stat().st_mtime)
        return {"type": "Poisson GLM", "version": mtime.strftime("%Y%m%d")}
    return {"type": "none", "version": ""}


def _load_predictions(target_date: date) -> pd.DataFrame:
    """
    Load saved feature/prediction parquet for target_date.

    Reads the local fast path first; if absent (e.g. this web container never
    ran the pipeline — the separate cron service did), falls back to object
    storage and caches the result locally. Returns empty DataFrame if neither
    source has it yet.
    """
    path = FEATURES_DIR / f"features_{target_date}.parquet"
    df: Optional[pd.DataFrame] = None

    if path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error(f"Failed to load local predictions for {target_date}: {exc}")
            df = None

    if df is None:
        # Cross-service handoff: pull what the cron service published.
        try:
            df = get_storage().get_parquet(features_key(target_date))
            if df is not None:
                df.to_parquet(path, index=False)  # cache locally for next read
                logger.info(f"Loaded predictions for {target_date} from storage")
        except Exception as exc:
            logger.error(f"Failed to load predictions from storage for {target_date}: {exc}")
            df = None

    if df is None or "predicted_k" not in df.columns:
        return pd.DataFrame()
    return df


def _df_to_predictions(df: pd.DataFrame) -> List[PredictionRow]:
    """Convert a features DataFrame to a list of PredictionRow objects."""
    if df.empty:
        return []

    rows = []
    for _, r in df.iterrows():
        def _f(col):
            v = r.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return round(float(v), 3)

        def _s(col, default=""):
            v = r.get(col)
            return str(v) if v is not None else default

        def _i(col, default=0):
            v = r.get(col)
            try:
                return int(v) if v == v else default
            except (TypeError, ValueError):
                return default

        is_home = bool(r.get("is_home", True))
        rows.append(PredictionRow(
            pitcher_name     = _s("pitcher_name", "Unknown"),
            pitcher_mlbam    = _i("pitcher_mlbam"),
            team             = _s("home_team") if is_home else _s("away_team"),
            opponent         = _s("away_team") if is_home else _s("home_team"),
            is_home          = is_home,
            predicted_k      = round(float(r.get("predicted_k", 0)), 1),
            lineup_confirmed = _i("lineup_confirmed"),
            model_version    = _s("model_version", _get_model_info()["version"]),
            k_rolling_5      = _f("k_rolling_5"),
            k_rolling_3      = _f("k_rolling_3"),
            whiff_pct_5      = _f("whiff_pct_5"),
            csw_pct_5        = _f("csw_pct_5"),
            opp_lineup_k_pct = _f("opp_lineup_k_pct"),
            opp_lineup_size  = _f("opp_lineup_size"),
            park_k_factor    = _f("park_k_factor"),
            days_rest        = _f("days_rest"),
            game_date        = _s("game_date")[:10] if r.get("game_date") else None,
            game_pk          = _i("game_pk") or None,
        ))

    # Sort descending by predicted_k
    rows.sort(key=lambda x: x.predicted_k, reverse=True)
    return rows


# Background pipeline refresh — one at a time
_refresh_lock = threading.Lock()
_refresh_running = False


def _run_pipeline_background(target_date: str):
    """Run pipeline.py in a subprocess (non-blocking)."""
    global _refresh_running
    if not _refresh_lock.acquire(blocking=False):
        logger.info("Pipeline refresh already running — skipping duplicate")
        return
    _refresh_running = True
    try:
        logger.info(f"Background pipeline refresh starting for {target_date}")
        result = subprocess.run(
            [sys.executable, "pipeline.py", "--date", target_date, "--no-wait"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → stdout so we get everything
            text=True,
            timeout=1200,  # 20-minute ceiling
            cwd=Path(__file__).parent,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        # Forward all pipeline output to Railway log stream
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[pipeline] {line}")
        if result.returncode == 0:
            logger.info(f"Pipeline refresh complete for {target_date}")
        else:
            logger.error(f"Pipeline refresh FAILED (exit {result.returncode}) for {target_date}")
    except subprocess.TimeoutExpired:
        logger.error("Pipeline refresh timed out after 10 minutes")
    except Exception as exc:
        logger.error(f"Pipeline refresh exception: {exc}")
    finally:
        _refresh_running = False
        _refresh_lock.release()


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root():
    """Redirect bare root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check API health, model version, and whether today's predictions exist."""
    model_info  = _get_model_info()
    today_df    = _load_predictions(date.today())
    return HealthResponse(
        status            = "ok",
        model_version     = model_info["version"],
        model_type        = model_info["type"],
        predictions_today = len(today_df),
        timestamp         = datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


@app.get("/predictions/today", response_model=List[PredictionRow], tags=["Predictions"])
def predictions_today():
    """
    Return today's pitcher predictions, sorted by projected strikeouts descending.

    If today's predictions haven't been computed yet, returns an empty list.
    Trigger a refresh with POST /admin/refresh.
    """
    df = _load_predictions(date.today())
    if df.empty:
        return []
    return _df_to_predictions(df)


@app.get("/predictions", response_model=List[PredictionRow], tags=["Predictions"])
def predictions_by_date(
    date: str = Query(
        ...,
        description="Game date in YYYY-MM-DD format (e.g. 2026-05-22)",
    )
):
    """
    Return predictions for a specific date.

    Useful for reviewing past predictions, backfilling the accuracy log,
    or checking historical matchups.
    """
    try:
        target = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")

    df = _load_predictions(target)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {date}. Run pipeline.py --date {date} to generate them.",
        )
    return _df_to_predictions(df)


@app.get("/metrics", response_model=MetricsSummary, tags=["Accuracy"])
def metrics(
    days: Optional[int] = Query(
        None,
        description="Rolling window in days (e.g. 30). Omit for all-time stats.",
    )
):
    """
    Return model accuracy summary: MAE, RMSE, bias, within-1-K rate.

    Split by confirmed vs projected lineup for diagnosing lineup data quality.
    """
    tracker = AccuracyTracker()
    summary = tracker.get_summary(window_days=days)
    return MetricsSummary(**summary)


@app.get("/metrics/trend", response_model=List[TrendRow], tags=["Accuracy"])
def metrics_trend(
    days: int = Query(
        90,
        description="How many days of history to return (default 90)",
    )
):
    """
    Return daily MAE and 7-day rolling MAE — feed this directly into a
    time-series chart in the UI to show model performance trend.

    Returns empty list if no graded predictions exist yet.
    """
    tracker = AccuracyTracker()
    df      = tracker.get_daily_mae(window_days=days)
    if df.empty:
        return []
    return [
        TrendRow(
            game_date      = str(r["game_date"]),
            n_graded       = int(r["n_graded"]),
            daily_mae      = float(r["daily_mae"]),
            rolling_mae_7d = float(r["rolling_mae_7d"]),
            bias           = float(r["bias"]),
        )
        for _, r in df.iterrows()
    ]


@app.post("/admin/refresh", response_model=RefreshResponse, tags=["Admin"])
def trigger_refresh(
    background_tasks: BackgroundTasks,
    date: Optional[str] = Query(
        None,
        description="Date to refresh (YYYY-MM-DD). Defaults to today.",
    ),
):
    """
    Trigger a background pipeline run to (re)generate predictions.

    Use this after lineups are confirmed (~1-3 PM ET on game days) to get
    updated predictions with real lineup data instead of projections.

    Only one refresh runs at a time — duplicate calls return immediately.
    """
    global _refresh_running
    target = date or datetime.today().strftime("%Y-%m-%d")

    try:
        datetime.strptime(target, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")

    if _refresh_running:
        return RefreshResponse(
            status  = "skipped",
            message = "A pipeline refresh is already running",
            date    = target,
        )

    background_tasks.add_task(_run_pipeline_background, target)
    return RefreshResponse(
        status  = "started",
        message = f"Pipeline refresh started for {target}. Results available in ~2-5 minutes.",
        date    = target,
    )


# ── Local dev entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
