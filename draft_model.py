"""Scores a prospect with the draft model exported by EPM_Draft.ipynb (last cell).

The app's form takes raw stats; the engineered and size-relative features are rebuilt here the same
way the notebook builds them. The notebook re-scores its board through this file after exporting and
prints a mismatch if the two drift, so keep DERIVED in step with the notebook's add_derived.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# display order for the form; a needed input not listed here goes at the end
RAW_INPUTS = [
    "ORtg", "usg", "twoP_per", "FT_per", "3par", "AST_per", "TO_per", "ast/tov",
    "ORB_per", "DRB_per", "blk_per", "stl_per",
    "ht_inches", "WEIGHT", "age", "projected_pick", "Rec Rank", "power6",
]

# engineered feature -> (raw columns it needs, recipe)
DERIVED = {
    "ORtgXUsage": (["usg", "ORtg"], lambda f: f["usg"] * f["ORtg"]),
    "2PTXUsage": (["usg", "twoP_per"], lambda f: f["usg"] * f["twoP_per"]),
    "ASTXUsage": (["usg", "AST_per"], lambda f: f["usg"] * f["AST_per"]),
    "stocks": (["blk_per", "stl_per"], lambda f: f["blk_per"] + f["stl_per"]),
    "Usg/Ast": (["usg", "AST_per"], lambda f: f["usg"] / f["AST_per"]),
}

GROUPS = ["guard", "wing", "big"]


def raw_inputs(features, size_rel):
    """Raw columns needed to build `features`. Height is always included (position fallback)."""
    need = []
    for feat in features:
        base = size_rel.get(feat, feat)
        for c in (DERIVED[base][0] if base in DERIVED else [base]):
            if c not in need:
                need.append(c)
    if "ht_inches" not in need:
        need.append("ht_inches")
    return [c for c in RAW_INPUTS if c in need] + [c for c in need if c not in RAW_INPUTS]


def pos_group(grp, ht):
    """Same rule as the notebook: the given group, else by height; a guard over 6'7" is a wing."""
    ht = np.nan if ht is None else float(ht)
    if grp not in GROUPS:
        grp = "guard" if ht < 79 else "wing" if ht < 82 else "big"
    if grp == "guard" and ht > 79:
        grp = "wing"
    return grp


def load(art_dir="artifacts"):
    """(bundle, booster), or None if the model hasn't been exported."""
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
    """raw: one row per player with the bundle's inputs + 'grp'. Returns X in FEATURES order."""
    f = raw.copy()
    for c in bundle["inputs"]:
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
        raise KeyError(f"draft_model.py can't build {missing}; add them to DERIVED")
    return f[bundle["features"]].to_numpy(dtype=np.float64)


def predict(raw: pd.DataFrame, bundle: dict, booster) -> np.ndarray:
    """Linear term on the mock pick (NaN -> training mean) + XGBoost residual."""
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
