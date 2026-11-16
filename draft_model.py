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

# Display order for the raw inputs the form asks for. Which ones it actually needs depends on the
# exported FEATURES (see raw_inputs below); a column not listed here is shown after these.
RAW_INPUTS = [
    "ORtg", "usg", "twoP_per", "FT_per", "3par", "AST_per", "TO_per", "ast/tov",
    "ORB_per", "DRB_per", "blk_per", "stl_per",
    "ht_inches", "WEIGHT", "age", "projected_pick", "power6",
]

# Engineered features, rebuilt from the raw inputs (same recipes as the notebook's model cell):
# name -> (the raw columns it needs, how to build it). Anything else in FEATURES is used as typed.
DERIVED = {
    "ORtgXUsage": (["usg", "ORtg"], lambda f: f["usg"] * f["ORtg"]),
    "2PTXUsage": (["usg", "twoP_per"], lambda f: f["usg"] * f["twoP_per"]),
    "ASTXUsage": (["usg", "AST_per"], lambda f: f["usg"] * f["AST_per"]),
    "stocks": (["blk_per", "stl_per"], lambda f: f["blk_per"] + f["stl_per"]),
    "Usg/Ast": (["usg", "AST_per"], lambda f: f["usg"] / f["AST_per"]),
}


def raw_inputs(features, size_rel):
    """The raw columns the form must ask for to rebuild `features` (the notebook's FEATURES after
    the _rel swap). Height is always included: the position group falls back to it."""
    need = []
    for feat in features:
        base = size_rel.get(feat, feat)
        for c in (DERIVED[base][0] if base in DERIVED else [base]):
            if c not in need:
                need.append(c)
    if "ht_inches" not in need:
        need.append("ht_inches")
    return [c for c in RAW_INPUTS if c in need] + [c for c in need if c not in RAW_INPUTS]

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
    """raw: one row per player with the bundle's raw inputs + 'grp'. Returns X in FEATURES order."""
    f = raw.copy()
    for c in (list(bundle.get("inputs", {})) or RAW_INPUTS):
        f[c] = pd.to_numeric(f[c], errors="coerce") if c in f.columns else np.nan
    for name, (deps, fn) in DERIVED.items():
        if all(c in f.columns for c in deps):
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
