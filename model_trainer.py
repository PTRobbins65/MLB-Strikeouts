"""
MLB Strikeout Prediction — Model Trainer
Trains XGBoost (Poisson objective) and a Poisson GLM on the full historical
feature matrix, evaluates both with time-series cross-validation, and saves
both models plus the winner label to models/.

Usage
-----
    python model_trainer.py                          # 2021–2024, 5-fold CV
    python model_trainer.py --start 2019 --end 2024
    python model_trainer.py --cv-folds 3 --refresh
"""

import argparse
import logging
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import FEATURES_DIR, LOG_DIR, MODEL_DIR
from data_fetcher import HistoricalDataFetcher
from feature_builder import FeatureBuilder
from game_log_builder import GameLogBuilder

logger = logging.getLogger("model_trainer")

TARGET_COL = "strikeouts"

# All features that feature_builder.build_row() can emit.
# Columns absent from a given run are silently skipped.
FEATURE_COLS = [
    # Pitcher rolling Statcast (4 windows × 6 metrics)
    *[f"k_rolling_{w}"    for w in [3, 5, 10, 20]],
    *[f"whiff_pct_{w}"    for w in [3, 5, 10, 20]],
    *[f"csw_pct_{w}"      for w in [3, 5, 10, 20]],
    *[f"zone_pct_{w}"     for w in [3, 5, 10, 20]],
    *[f"o_swing_pct_{w}"  for w in [3, 5, 10, 20]],
    *[f"avg_velo_{w}"     for w in [3, 5, 10, 20]],
    # FanGraphs season-level
    "k9_season", "k_pct_season", "bb_pct_season",
    "fip_season", "xfip_season", "swstr_season", "stuff_plus",
    # Workload
    "days_rest", "pitches_last",
    # Opponent lineup
    "opp_lineup_k_pct", "opp_lineup_o_swing", "opp_lineup_contact",
    "opp_lineup_swstr", "opp_lineup_wrc_plus", "opp_lineup_size",
    # Context
    "park_k_factor", "is_home", "is_night_game",
    "lineup_confirmed", "umpire_k_pct",
]


class ModelTrainer:
    """
    Trains, cross-validates, and saves the strikeout prediction models.

    Parameters
    ----------
    cv_folds : number of time-series folds for cross-validation
    """

    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.fetcher  = HistoricalDataFetcher()

    # ── Feature matrix construction ────────────────────────────────────────

    def build_feature_matrix(
        self,
        game_log: pd.DataFrame,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """Load supporting data and run FeatureBuilder over the full game log."""
        logger.info("Loading FanGraphs pitcher and batter stats...")
        fg_pitchers = self.fetcher.get_fangraphs_stats(start_year, end_year)
        fg_batters  = self.fetcher.get_fangraphs_batter_stats(start_year, end_year)

        start_dt = f"{start_year}-03-01"
        end_dt   = f"{end_year}-11-01"
        logger.info("Loading league-wide Statcast per-start metrics...")
        pitches = self.fetcher.get_statcast_date_range(start_dt, end_dt)
        if pitches.empty:
            raise RuntimeError("No Statcast data available for the training window")
        statcast_starts = self.fetcher.compute_per_start_metrics(pitches)

        builder = FeatureBuilder(
            fg_pitcher_df   = fg_pitchers,
            fg_batter_df    = fg_batters,
            statcast_starts = statcast_starts,
        )

        # Build historical lineup lookup from Statcast so training rows have
        # real opponent batters rather than empty synthetic LineupCards.
        logger.info("Building historical lineup lookup from Statcast data...")
        game_meta = (
            game_log[["game_pk", "game_date", "home_team", "away_team",
                       "home_team_id", "away_team_id"]]
            .drop_duplicates("game_pk")
        )
        lineup_lookup = self.fetcher.build_lineup_lookup(pitches, game_meta=game_meta)

        logger.info(f"Building feature matrix for {len(game_log):,} pitcher-starts...")
        features_df = builder.build_training_set(game_log, lineup_lookup=lineup_lookup)
        logger.info(
            f"Feature matrix: {len(features_df):,} rows × {len(features_df.columns)} cols"
        )
        return features_df

    # ── Cross-validation ───────────────────────────────────────────────────

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        name: str,
    ) -> dict:
        """Time-series k-fold CV. Returns mean/std of MAE and RMSE."""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        maes, rmses = [], []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
            y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            preds = np.clip(model.predict(X_val), 0, None)

            mae  = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            maes.append(mae);  rmses.append(rmse)
            logger.info(f"  {name} fold {fold}/{self.cv_folds}: MAE={mae:.3f}  RMSE={rmse:.3f}")

        result = {
            "model_name": name,
            "mean_mae":   float(np.mean(maes)),
            "std_mae":    float(np.std(maes)),
            "mean_rmse":  float(np.mean(rmses)),
            "std_rmse":   float(np.std(rmses)),
        }
        logger.info(
            f"  {name} — MAE {result['mean_mae']:.3f} ± {result['std_mae']:.3f}  |  "
            f"RMSE {result['mean_rmse']:.3f} ± {result['std_rmse']:.3f}"
        )
        return result

    # ── Train and save ─────────────────────────────────────────────────────

    def train(self, features_df: pd.DataFrame, save: bool = True) -> dict:
        """
        Train both models on features_df, compare CV metrics, save all artifacts.
        Returns a summary dict with CV results and which model won.
        """
        avail = [c for c in FEATURE_COLS if c in features_df.columns]
        missing = set(FEATURE_COLS) - set(avail)
        if missing:
            logger.warning(f"{len(missing)} feature cols not found (will be absent): {sorted(missing)}")

        X = features_df[avail].copy()
        y = features_df[TARGET_COL].copy()

        # Sort chronologically to prevent leakage in TimeSeriesSplit
        if "game_date" in features_df.columns:
            order = pd.to_datetime(features_df["game_date"]).argsort()
            X = X.iloc[order].reset_index(drop=True)
            y = y.iloc[order].reset_index(drop=True)

        logger.info(
            f"Training on {len(X):,} samples | {len(avail)} features | "
            f"{self.cv_folds}-fold time-series CV"
        )

        # ── Model definitions ──────────────────────────────────────────────
        xgb_model = xgb.XGBRegressor(
            objective        = "count:poisson",
            n_estimators     = 500,
            learning_rate    = 0.05,
            max_depth        = 4,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            random_state     = 42,
            n_jobs           = -1,
        )

        glm_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("glm",     PoissonRegressor(alpha=0.1, max_iter=500)),
        ])

        # ── Cross-validate ─────────────────────────────────────────────────
        logger.info("── XGBoost (Poisson) cross-validation ──")
        xgb_cv = self._cross_validate(X, y, xgb_model, "XGBoost-Poisson")

        logger.info("── Poisson GLM cross-validation ──")
        glm_cv = self._cross_validate(X, y, glm_model, "Poisson-GLM")

        # ── Final fit on full data ─────────────────────────────────────────
        logger.info("Fitting final models on full training set...")
        xgb_model.fit(X, y)
        glm_model.fit(X, y)

        # ── Feature importances (XGBoost) ──────────────────────────────────
        importances = pd.Series(
            xgb_model.feature_importances_, index=avail
        ).sort_values(ascending=False)
        logger.info("Top-15 feature importances (XGBoost gain):")
        for feat, imp in importances.head(15).items():
            logger.info(f"  {feat:<40} {imp:.4f}")

        winner = "xgb" if xgb_cv["mean_mae"] <= glm_cv["mean_mae"] else "glm"
        summary = {
            "xgb_cv":   xgb_cv,
            "glm_cv":   glm_cv,
            "winner":   winner,
            "features": avail,
            "n_train":  len(X),
        }

        # ── Persist models ─────────────────────────────────────────────────
        if save:
            xgb_path  = MODEL_DIR / "strikeout_xgb.json"
            glm_path  = MODEL_DIR / "strikeout_glm.joblib"
            feat_path = MODEL_DIR / "feature_importances.parquet"
            meta_path = MODEL_DIR / "model_meta.parquet"

            xgb_model.save_model(str(xgb_path))
            joblib.dump(glm_model, glm_path)
            importances.to_frame("importance").to_parquet(feat_path)

            meta = {
                "winner":      winner,
                "features":    str(avail),
                "n_train":     len(X),
                "xgb_mae":     xgb_cv["mean_mae"],
                "glm_mae":     glm_cv["mean_mae"],
                "cv_folds":    self.cv_folds,
            }
            pd.DataFrame([meta]).to_parquet(meta_path, index=False)

            logger.info(f"XGBoost saved  → {xgb_path.name}")
            logger.info(f"Poisson GLM saved → {glm_path.name}")
            logger.info(f"Winner: {'XGBoost' if winner == 'xgb' else 'Poisson GLM'}")

        return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MLB strikeout prediction models")
    p.add_argument("--start",    type=int, default=2021, help="First season (default 2021)")
    p.add_argument("--end",      type=int, default=2024, help="Last season (default 2024)")
    p.add_argument("--cv-folds", type=int, default=5,    help="CV folds (default 5)")
    p.add_argument("--refresh",  action="store_true",    help="Force re-fetch all data")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "model_trainer.log"),
        ],
    )

    args = parse_args()

    # 1. Build game log
    log_builder = GameLogBuilder()
    game_log    = log_builder.build(args.start, args.end, force_refresh=args.refresh)
    if game_log.empty:
        logger.error("Game log is empty — run game_log_builder.py first or check data availability")
        sys.exit(1)
    logger.info(f"Game log: {len(game_log):,} pitcher-starts ({args.start}–{args.end})")

    # 2. Build feature matrix
    trainer     = ModelTrainer(cv_folds=args.cv_folds)
    features_df = trainer.build_feature_matrix(game_log, args.start, args.end)
    if features_df.empty:
        logger.error("Feature matrix is empty — check FanGraphs and Statcast availability")
        sys.exit(1)

    # 3. Train, evaluate, and save
    summary = trainer.train(features_df, save=True)

    print("\n── Training Summary ──────────────────────────────")
    print(f"  Samples trained on : {summary['n_train']:,}")
    print(f"  Features used      : {len(summary['features'])}")
    print(f"  XGBoost  MAE       : {summary['xgb_cv']['mean_mae']:.3f} ± {summary['xgb_cv']['std_mae']:.3f}")
    print(f"  Poisson GLM MAE    : {summary['glm_cv']['mean_mae']:.3f} ± {summary['glm_cv']['std_mae']:.3f}")
    print(f"  Winner             : {'XGBoost' if summary['winner'] == 'xgb' else 'Poisson GLM'}")
    print(f"  Models saved to    : {MODEL_DIR}")
