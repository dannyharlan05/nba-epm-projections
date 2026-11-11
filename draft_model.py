"""Scores a prospect with the exported draft model (Darko_Draft.ipynb -> RANKINGS PAGE cell).

The form on the Draft Rankings page takes RAW stats only. Everything the model actually sees --
interactions (ORtg x usage, ...), stocks, Usg/Ast and the size-relative "_rel" versions -- is
rebuilt here from those raw inputs, exactly the way the notebook builds them. The notebook imports
this file after exporting and checks it reproduces the board, so if a feature recipe changes there,
change DERIVED here too (the export cell will print a mismatch if they drift).

Artifacts (written by the notebook):
  artifacts/draft_model.json  -- FEATURES order, size medians, position mask, caps, linear base, input stats
  artifacts/draft_xgb.json    -- the XGBoost residual model
  artifacts/draft_inputs.csv  -- raw inputs for every board player (to prefill the form)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# Raw inputs the form asks for, in display order.
RAW_INPUTS = [
    "ORtg", "usg", "twoP_per", "FT_per", "AST_per", "TO_per",
    "ORB_per", "DRB_per", "blk_per", "stl_per",
    "ht_inches", "WEIGHT", "age", "projected_pick", "power6",
]

# Engineered features, rebuilt from the raw inputs (same recipes as the notebook's model cell).
DERIVED = {
    "ORtgXUsage": lambda f: f["usg"] * f["ORtg"],
    "2PTXUsage": lambda f: f["usg"] * f["twoP_per"],
    "ASTXUsage": lambda f: f["usg"] * f["AST_per"],
    "stocks": lambda f: f["blk_per"] + f["stl_per"],
    "Usg/Ast": lambda f: f["usg"] / f["AST_per"],
}

GROUPS = ["guard", "wing", "big"]


def pos_group(grp, ht):
    """Guard/wing/big, same rule as the notebook's _pos_group: the given group (from the
    Barttorvik role) if any, else by height; a 'guard' over 6'7" counts as a wing."""
    ht = np.nan if ht is None else float(ht)
    if grp not in GROUPS:
        grp = "guard" if ht < 79 else "wing" if ht < 82 else "big"   # NaN height -> big, as in the notebook
    if grp == "guard" and ht > 79:
        grp = "wing"
    return grp


def load(art_dir="artifacts"):
    """(bundle, booster) or None if the notebook hasn't exported the model yet."""
    meta, xgb_path = os.path.join(art_dir, "draft_model.json"), os.path.join(art_dir, "draft_xgb.json")
    if not (os.path.exists(meta) and os.path.exists(xgb_path)):
        return None
    import xgboost as xgb
    with open(meta) as fh:
        bundle = json.load(fh)
    booster = xgb.Booster()
    booster.load_model(xgb_path)
    return bundle, booster


def feature_matrix(raw: pd.DataFrame, bundle: dict) -> np.ndarray:
    """raw: one row per player with RAW_INPUTS + 'grp'. Returns X in the model's FEATURES order."""
    f = raw.copy()
    for c in RAW_INPUTS:
        f[c] = pd.to_numeric(f[c], errors="coerce") if c in f.columns else np.nan
    for name, fn in DERIVED.items():
        f[name] = fn(f)
    f = f.replace([np.inf, -np.inf], np.nan)
    grp = f["grp"].astype(str)
    for new, base in bundle["size_rel"].items():
        f[new] = f[base] - grp.map(bundle["size_med"][new]).astype(float)
    for feat, drops in bundle.get("position_mask", {}).items():
        if feat in f.columns:
            f[feat] = f[feat].where(~grp.isin(drops))
    for feat, hi in bundle.get("feature_caps", {}).items():
        if feat in f.columns:
            f[feat] = f[feat].clip(upper=hi)
    missing = [c for c in bundle["features"] if c not in f.columns]
    if missing:
        raise KeyError(f"draft_model.py can't build {missing}; add them to RAW_INPUTS / DERIVED")
    return f[bundle["features"]].to_numpy(dtype=np.float64)


def predict(raw: pd.DataFrame, bundle: dict, booster) -> np.ndarray:
    """Predicted peak EPM = linear base on LIN_FEATURES (NaN -> training mean) + XGBoost residual."""
    import xgboost as xgb
    X = feature_matrix(raw, bundle)
    base = np.zeros(len(X))
    lin = bundle.get("lin")
    if lin:
        Z = X[:, lin["idx"]].copy()
        for j, fv in enumerate(lin["fill"]):
            Z[np.isnan(Z[:, j]), j] = fv
        base = Z @ np.asarray(lin["coef"], float) + lin["intercept"]
    return base + booster.predict(xgb.DMatrix(X, missing=np.nan))
