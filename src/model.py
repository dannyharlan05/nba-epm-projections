"""EPM projection models: leak-free temporal CV, median + p90 ceiling, OOF predictions.

Single source of truth for hyperparameters: DEFAULT_PARAMS + PARAMS_BY_HORIZON.
Edit those to retune every horizon; everything downstream reads from them.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

HORIZONS = [1, 2, 3, 4, 5]

# Features: EPM-centric, with DARKO (dpm) and actual-EPM signals as cross-features.
EPM_FEATURES = [
    "epm_now", "epm_lag1", "epm_lag2",
    "age", "years_in_league",
    "min_pg", "young_improver",
    "total_minutes", "minutes_x_games",
    "epm_x_games", "dpm_x_games", "epm_actual_x_games",
    "epm_x_min_pg", "dpm_x_min_pg", "epm_actual_x_min_pg",
    "pts_36", "ast_36", "oreb_36", "stl_36", "blk_36",
    "ft_pct", "draft_score", "team_winpct",
    "dpm",
    "epm_actual_now", "epm_actual_lag1", "epm_actual_lag2",
]

# Features for the actual (observed) EPM model. Identical for now; edit independently.
EPM_ACTUAL_FEATURES = list(EPM_FEATURES)

# Monotonic +1: more current EPM never lowers the projection (ceteris paribus).
_MONO_UP = {"epm_now"}
_MONO = tuple(1 if f in _MONO_UP else 0 for f in EPM_FEATURES)

DEFAULT_PARAMS = dict(
    n_estimators=800, learning_rate=0.02, max_depth=4,
    subsample=0.6, colsample_bytree=1.0,
    random_state=42, n_jobs=-1, monotone_constraints=_MONO,
)

# Per-horizon overrides; {} means "use DEFAULT_PARAMS".
PARAMS_BY_HORIZON = {
    1: {"learning_rate": 0.01},
    2: {"learning_rate": 0.01},
    3: {"learning_rate": 0.01},
    4: {"learning_rate": 0.01},
    5: {"learning_rate": 0.01},
}


def params_for(horizon: int) -> dict:
    return {**DEFAULT_PARAMS, **PARAMS_BY_HORIZON.get(horizon, {})}


# Recency weighting: recent seasons count more, so the aging curve reflects the modern
# game (longer careers) rather than being dragged down by early-2000s washouts. Weights
# are anchored to the max season of whatever set they're fit on, so a CV fold never
# references anything outside its own training rows (leak-free by construction).
# Smaller HALF_LIFE = more aggressive recency bias. None = off (equal weights).
HALF_LIFE = 6


def recency_weights(seasons) -> np.ndarray:
    seasons = np.asarray(seasons, dtype=float)
    if HALF_LIFE is None:
        return np.ones(len(seasons))
    return 0.5 ** ((seasons.max() - seasons) / HALF_LIFE)


def temporal_splits(seasons: pd.Series, n_test_seasons: int = 2):
    """CV folds where test seasons are always strictly after all training seasons."""
    uniq = sorted(seasons.unique())
    n = len(uniq)
    min_train = max(5, n // 3)
    splits = []
    for test_end in range(min_train + n_test_seasons, n + n_test_seasons, n_test_seasons):
        test_start = test_end - n_test_seasons
        train = set(uniq[:test_start])
        test = set(uniq[test_start:test_end])
        splits.append((np.where(seasons.isin(train))[0],
                       np.where(seasons.isin(test))[0]))
    return splits


def cv_report(df: pd.DataFrame, features=EPM_FEATURES, target_prefix="target_epm"):
    """Train vs CV MAE per horizon (overfitting check)."""
    rows = []
    for h in HORIZONS:
        tgt = f"{target_prefix}_{h}y"
        dfh = df.dropna(subset=[tgt])
        X, y = dfh[features].values, dfh[tgt].values
        seasons = dfh["season"].values
        p = params_for(h)
        tr_maes, te_maes = [], []
        for tr, te in temporal_splits(dfh["season"]):
            m = xgb.XGBRegressor(**p)
            m.fit(X[tr], y[tr], sample_weight=recency_weights(seasons[tr]))
            tr_maes.append(mean_absolute_error(y[tr], m.predict(X[tr])))
            te_maes.append(mean_absolute_error(y[te], m.predict(X[te])))
        rows.append({"horizon": h, "train_mae": np.mean(tr_maes),
                     "cv_mae": np.mean(te_maes), "gap": np.mean(te_maes) - np.mean(tr_maes)})
    return pd.DataFrame(rows)


def train_models(df: pd.DataFrame, features=EPM_FEATURES, target_prefix="target_epm",
                 quantile_alpha: float | None = None) -> dict:
    """Fit one model per horizon on all rows with a target.
    quantile_alpha set (e.g. 0.90) -> ceiling/quantile models."""
    models = {}
    for h in HORIZONS:
        tgt = f"{target_prefix}_{h}y"
        dfh = df.dropna(subset=[tgt])
        p = params_for(h)
        if quantile_alpha is not None:
            p = {**p, "objective": "reg:quantileerror", "quantile_alpha": quantile_alpha}
        m = xgb.XGBRegressor(**p)
        m.fit(dfh[features].values, dfh[tgt].values,
              sample_weight=recency_weights(dfh["season"].values))
        models[h] = m
    return models


def add_oof_predictions(df: pd.DataFrame, features=EPM_FEATURES,
                        target_prefix="target_epm", out_prefix="pred_epm",
                        quantile_alpha: float | None = None) -> pd.DataFrame:
    """Leak-free predictions: OOF for rows with a target, final model for current
    (no-target) rows. Returns df with out_prefix_{h}y columns added."""
    df = df.copy()
    for h in HORIZONS:
        tgt = f"{target_prefix}_{h}y"
        dfh = df.dropna(subset=[tgt]).copy()
        X, y = dfh[features].values, dfh[tgt].values
        seasons = dfh["season"].values
        p = params_for(h)
        if quantile_alpha is not None:
            p = {**p, "objective": "reg:quantileerror", "quantile_alpha": quantile_alpha}

        oof = np.full(len(dfh), np.nan)
        in_oof = np.zeros(len(dfh), dtype=bool)
        m = xgb.XGBRegressor(**p)
        for tr, te in temporal_splits(dfh["season"]):
            m.fit(X[tr], y[tr], sample_weight=recency_weights(seasons[tr]))
            oof[te] = m.predict(X[te])
            in_oof[te] = True

        col = f"{out_prefix}_{h}y"
        df[col] = np.nan
        df.loc[dfh.index[in_oof], col] = oof[in_oof]

        m.fit(X, y, sample_weight=recency_weights(seasons))  # final model, current season
        no_t = df[tgt].isna()
        df.loc[no_t, col] = m.predict(df.loc[no_t, features].values)
    return df
