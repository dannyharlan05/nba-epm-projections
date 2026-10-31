"""Data loading for the EPM projection pipeline.

All raw inputs live in ./data. The two NBA-stat inputs (game logs, DARKO) and the
two EPM inputs (Dunks & Threes predictive + actual) are CSVs you drop in there;
draft history is pulled once from the NBA API and cached to data/draft_history.csv.
"""
from __future__ import annotations
import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("EPM_DATA_DIR", "data")

def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def last_complete_season(today=None) -> int:
    """Seasons are named by end year (2026 = 2025-26); one counts as finished from
    July 1 of that year, so the season in progress never leaks into training."""
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    return today.year if today.month >= 7 else today.year - 1


def _newest_per_season(pattern: str, through: int | None = None) -> pd.DataFrame:
    """Every CSV matching `pattern`; for each season, the rows from the most recently
    downloaded file (file modified time). File names don't matter, so browser download
    numbers like '(29)' or a re-downloaded history are handled the same on any machine.
    Seasons after `through` (default: last completed season) are dropped."""
    through = last_complete_season() if through is None else through
    files = glob.glob(_path(pattern))
    if not files:
        raise FileNotFoundError(f"no '{pattern}' files in {DATA_DIR}")
    df = pd.concat([pd.read_csv(f).assign(_mtime=os.path.getmtime(f)) for f in files],
                   ignore_index=True)
    df = df.dropna(subset=["season"])
    df = df[df["season"].astype(int) <= through]
    df = df[df["_mtime"] == df.groupby("season")["_mtime"].transform("max")]
    return df.drop(columns="_mtime")


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


def load_predictive_epm(through: int | None = None) -> pd.DataFrame:
    """Dunks & Threes *predictive* EPM ('EPM data*.csv'), newest download per season."""
    epm = _newest_per_season("EPM data*.csv", through)
    epm = epm[["season", "nba_id", "oepm", "depm", "epm", "p_usg"]].dropna(subset=["nba_id"])
    epm["nba_id"] = epm["nba_id"].astype(int)
    epm["season"] = epm["season"].astype(int)
    return epm.drop_duplicates(["nba_id", "season"])


def load_actual_epm(through: int | None = None) -> pd.DataFrame | None:
    """Dunks & Threes *actual* (observed) EPM ('Dunks & Threes Stats*.csv'), newest
    download per season. Returns None if the files are absent."""
    try:
        act = _newest_per_season("Dunks & Threes Stats*.csv", through)
    except FileNotFoundError:
        return None
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
