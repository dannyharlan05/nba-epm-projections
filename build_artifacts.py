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
    draft_scores = data.load_draft_scores()

    print("building training table...")
    df = features.build_training_table(darko, draft, logs, epm, actual, draft_scores)
    print(f"  df_train: {df.shape}")

    print("cross-validation (median)...")
    cv = model.cv_report(df)
    print(cv.to_string(index=False))

    print("generating leak-free predictions...")
    df = model.add_oof_predictions(df, out_prefix="pred_epm")

    print("fitting final models...")
    models = model.train_models(df)

    print("saving artifacts...")
    df.to_parquet(os.path.join(ART, "predictions.parquet"))
    with open(os.path.join(ART, "epm_models.pkl"), "wb") as f:
        pickle.dump(models, f)
    with open(os.path.join(ART, "meta.json"), "w") as f:
        json.dump({
            "features": model.EPM_FEATURES,
            "horizons": model.HORIZONS,
            "default_params": {k: v for k, v in model.DEFAULT_PARAMS.items()
                               if k != "monotone_constraints"},
            "cv_mae": cv.to_dict(orient="records"),
            "current_season": int(df["season"].max()),
        }, f, indent=2)

    print("done. artifacts written to ./artifacts")


if __name__ == "__main__":
    main()
