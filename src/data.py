"""Data loading for the EPM projection pipeline.

All raw inputs live in ./data. The two NBA-stat inputs (game logs, DARKO) and the
two EPM inputs (Dunks & Threes predictive + actual) are CSVs you drop in there;
draft history is pulled once from the NBA API and cached to data/draft_history.csv.
"""
from __future__ import annotations
import glob
import os
import re

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("EPM_DATA_DIR", "data")


def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def load_game_logs() -> pd.DataFrame:
    """Per-game player logs (one row per player-game)."""
    return pd.read_csv(_path("player_game_logs_clean.csv"))


def load_darko() -> pd.DataFrame:
    """DARKO DPM history (box-score-prior daily-adjusted plus-minus)."""
    return pd.read_csv(
        _path("DARKO - Daily Adjusted and Regressed Kalman Optimized "
              "projections - Full DPM History.csv")
    )


def load_draft(refresh: bool = False) -> pd.DataFrame:
    """Draft history. Cached to data/draft_history.csv; pulled from NBA API if missing.

    Pass refresh=True to force a re-pull.
    """
    cache = _path("draft_history.csv")
    if os.path.exists(cache) and not refresh:
        draft = pd.read_csv(cache)
    else:
        from nba_api.stats.endpoints import drafthistory
        draft = drafthistory.DraftHistory().get_data_frames()[0]
        draft = draft[[
            "PERSON_ID", "PLAYER_NAME", "SEASON", "ROUND_NUMBER",
            "ROUND_PICK", "OVERALL_PICK", "TEAM_CITY",
        ]].rename(columns={"PERSON_ID": "nba_id", "SEASON": "draft_year"})
        draft.to_csv(cache, index=False)

    draft["ROUND_NUMBER"] = draft["ROUND_NUMBER"].fillna(0).astype(int)
    draft["OVERALL_PICK"] = draft["OVERALL_PICK"].fillna(999).astype(int)
    draft["is_first_round"] = (draft["ROUND_NUMBER"] == 1).astype(int)
    return draft


def load_predictive_epm(min_file: int = 2) -> pd.DataFrame:
    """Dunks & Threes *predictive* EPM ('EPM data (N).csv'), N >= min_file."""
    def _filenum(f):
        m = re.search(r"\((\d+)\)", f)
        return int(m.group(1)) if m else 0

    files = [f for f in glob.glob(_path("EPM data*.csv")) if _filenum(f) >= min_file]
    if not files:
        raise FileNotFoundError(f"no 'EPM data (>= {min_file}).csv' files in {DATA_DIR}")
    epm = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    epm = epm[["season", "nba_id", "oepm", "depm", "epm", "p_usg"]].dropna(subset=["nba_id"])
    epm["nba_id"] = epm["nba_id"].astype(int)
    epm["season"] = epm["season"].astype(int)
    return epm.drop_duplicates(["nba_id", "season"])


def load_actual_epm() -> pd.DataFrame | None:
    """Dunks & Threes *actual* (observed) EPM. Returns None if files absent."""
    files = glob.glob(_path("Dunks & Threes Stats*.csv"))
    if not files:
        return None
    act = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    act = act[act["seasontype"] == 2]  # regular season
    act = act.rename(columns={
        "player_id": "nba_id", "off": "oepm_actual",
        "def": "depm_actual", "tot": "epm_actual_now",
    })
    act = act[["season", "nba_id", "epm_actual_now", "oepm_actual", "depm_actual"]]
    act = act.dropna(subset=["nba_id"])
    act["nba_id"] = act["nba_id"].astype(int)
    act["season"] = act["season"].astype(int)
    return act.drop_duplicates(["nba_id", "season"])
