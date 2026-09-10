"""Build everything the app needs, then save to ./artifacts.

Run once after dropping the raw CSVs in ./data:

    python build_artifacts.py

Outputs:
    artifacts/predictions.parquet   df_train with pred_epm_*
    artifacts/epm_models.pkl        models (one per horizon)
    artifacts/meta.json             features, params, CV MAE table
"""
from __future__ import annotations
import json
import os
import pickle

import pandas as pd

from src import data, features, model

ART = "artifacts"


def main():
    os.makedirs(ART, exist_ok=True)

    print("loading raw data...")
    darko = data.load_darko()
    draft = data.load_draft()
    logs = data.load_game_logs()
    epm = data.load_predictive_epm()
    actual = data.load_actual_epm()

    print("building training table...")
    df = features.build_training_table(darko, draft, logs, epm, actual)
    print(f"  df_train: {df.shape}")

    print("cross-validation (predictive EPM)...")
    cv = model.cv_report(df)
    print(cv.to_string(index=False))

    print("cross-validation (actual EPM)...")
    cv_actual = model.cv_report(df, features=model.EPM_ACTUAL_FEATURES,
                                target_prefix="target_epm_actual")
    print(cv_actual.to_string(index=False))

    print("generating out-of-fold predictions (predictive + actual)...")
    df = model.add_oof_predictions(df, out_prefix="pred_epm")
    df = model.add_oof_predictions(df, features=model.EPM_ACTUAL_FEATURES,
                                   target_prefix="target_epm_actual",
                                   out_prefix="pred_epm_actual")

    print("fitting final models...")
    models = model.train_models(df)
    models_actual = model.train_models(df, features=model.EPM_ACTUAL_FEATURES,
                                       target_prefix="target_epm_actual")

    # Players who sat out the whole current season are dropped from df (no game
    # logs), so the app would simply not list them. Build inference-only ghost
    # rows for them and project with the final models. Nothing here is fitted on.
    print("building ghost rows (played last season, sat out this one)...")
    ghosts = features.ghost_rows(df, darko, epm)
    if len(ghosts):
        for h in model.HORIZONS:
            ghosts[f"pred_epm_{h}y"] = models[h].predict(
                ghosts[model.EPM_FEATURES].values.astype(float))
            ghosts[f"pred_epm_actual_{h}y"] = models_actual[h].predict(
                ghosts[model.EPM_ACTUAL_FEATURES].values.astype(float))
        src_n = int((ghosts["level_source"] == "source").sum())
        print(f"  {len(ghosts)} ghosts ({src_n} with a live DARKO/EPM level, "
              f"{len(ghosts) - src_n} carried forward)")
    df["status"] = "played"
    if len(ghosts):
        ghosts["status"] = f"missed {int(df['season'].max())}"
        df = pd.concat([df, ghosts[[c for c in ghosts.columns if c in df.columns]]],
                       ignore_index=True)

    print("saving artifacts...")
    # only the columns the app reads -- the full training table is ~8x the size
    # and exposes targets the app has no use for
    app_cols = [c for c in [
        "player_name", "season", "age", "team", "epm_now", "epm_actual_now",
        *[f"pred_epm_{h}y" for h in model.HORIZONS],
        *[f"pred_epm_actual_{h}y" for h in model.HORIZONS],
        "status",
    ] if c in df.columns]
    df[app_cols].to_parquet(os.path.join(ART, "predictions.parquet"))
    with open(os.path.join(ART, "epm_models.pkl"), "wb") as f:
        pickle.dump(models, f)
    with open(os.path.join(ART, "epm_actual_models.pkl"), "wb") as f:
        pickle.dump(models_actual, f)
    with open(os.path.join(ART, "meta.json"), "w") as f:
        json.dump({
            "features": model.EPM_FEATURES,
            "horizons": model.HORIZONS,
            "default_params": {k: v for k, v in model.DEFAULT_PARAMS.items()
                               if k != "monotone_constraints"},
            "cv_mae": cv.to_dict(orient="records"),
            "cv_mae_actual": cv_actual.to_dict(orient="records"),
            "current_season": int(df.loc[df["status"] == "played", "season"].max()),
        }, f, indent=2)

    print("done. artifacts written to ./artifacts")


if __name__ == "__main__":
    main()
