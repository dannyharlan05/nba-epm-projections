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

    print("saving artifacts...")
    df.to_parquet(os.path.join(ART, "predictions.parquet"))
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
            "current_season": int(df["season"].max()),
        }, f, indent=2)

    print("done. artifacts written to ./artifacts")


if __name__ == "__main__":
    main()
