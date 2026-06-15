"""Feature engineering: build the per-(player, season) training table `df_train`.

Pipeline order (each function takes and returns df_train):
    build_base_dataset -> merge_draft_scores -> add_team_context
    -> add_epm_features -> apply_survivorship -> add_availability_features
    -> add_young_improver

Keys: nba_id (int), season (int, end-year convention e.g. 2024-25 -> 2025).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_base_dataset(darko: pd.DataFrame, draft: pd.DataFrame,
                       logs: pd.DataFrame) -> pd.DataFrame:
    """Season-level box features from game logs, merged with DARKO + draft.

    Also builds DPM lags/deltas/slope and the forward DPM targets.
    """
    logs = logs[logs["MIN"] >= 10].copy()

    for stat in ["PTS", "AST", "OREB", "DREB", "STL", "BLK", "TOV", "FGA", "FG3A", "FTA"]:
        logs[f"{stat}_36"] = logs[stat] / logs["MIN"] * 36
    logs["FT_PCT"] = logs["FTM"] / logs["FTA"].replace(0, np.nan)
    logs["FG3_PCT"] = logs["FG3M"] / logs["FG3A"].replace(0, np.nan)
    logs["TS_PCT"] = logs["PTS"] / (2 * (logs["FGA"] + 0.44 * logs["FTA"])).replace(0, np.nan)

    # box-score rates averaged over ALL games (regular season + playoffs)
    season_stats = logs.groupby(["PLAYER_ID", "SEASON_YEAR"]).agg(
        min_pg=("MIN", "mean"),
        pts_36=("PTS_36", "mean"), ast_36=("AST_36", "mean"),
        oreb_36=("OREB_36", "mean"), dreb_36=("DREB_36", "mean"),
        stl_36=("STL_36", "mean"), blk_36=("BLK_36", "mean"),
        tov_36=("TOV_36", "mean"), fg3a_36=("FG3A_36", "mean"),
        fta_36=("FTA_36", "mean"), ft_pct=("FT_PCT", "mean"),
        fg3_pct=("FG3_PCT", "mean"), ts_pct=("TS_PCT", "mean"),
    ).reset_index()

    # games = REGULAR-SEASON count only (GAME_ID 3rd digit == '2'), so it's a true
    # 0-82 durability measure; the rate stats above still include playoffs.
    _gtype = logs["GAME_ID"].astype(str).str.zfill(10).str[2]
    _reg = (logs[_gtype == "2"].groupby(["PLAYER_ID", "SEASON_YEAR"])
            .size().reset_index(name="games"))
    season_stats = season_stats.merge(_reg, on=["PLAYER_ID", "SEASON_YEAR"], how="left")
    season_stats["games"] = season_stats["games"].fillna(0)

    season_stats = season_stats.rename(columns={"PLAYER_ID": "nba_id", "SEASON_YEAR": "season"})
    season_stats["nba_id"] = season_stats["nba_id"].astype(int)
    season_stats["season"] = season_stats["season"].astype(str).str[-2:].astype(int) + 2000

    df = darko.merge(
        draft[["nba_id", "OVERALL_PICK", "ROUND_NUMBER", "is_first_round", "draft_year"]],
        on="nba_id", how="left")
    df = df.merge(season_stats, on=["nba_id", "season"], how="left")

    df["OVERALL_PICK"] = df["OVERALL_PICK"].fillna(999)
    df["ROUND_NUMBER"] = df["ROUND_NUMBER"].fillna(0)
    df["is_first_round"] = df["is_first_round"].fillna(0)
    df["draft_year"] = df["draft_year"].astype(str).str[:4].astype(float)
    df["years_in_league"] = df["season"] - df["draft_year"]

    df = df.sort_values(["nba_id", "season"])
    for lag in [1, 2]:
        df[f"dpm_lag{lag}"] = df.groupby("nba_id")["dpm"].shift(lag)
        df[f"o_dpm_lag{lag}"] = df.groupby("nba_id")["o_dpm"].shift(lag)
        df[f"d_dpm_lag{lag}"] = df.groupby("nba_id")["d_dpm"].shift(lag)
    df["dpm_delta1"] = df["dpm"] - df["dpm_lag1"]
    df["dpm_delta2"] = df["dpm_lag1"] - df["dpm_lag2"]

    def _slope(row):
        vals = [row["dpm_lag2"], row["dpm_lag1"], row["dpm"]]
        if any(pd.isna(v) for v in vals):
            return np.nan
        return np.polyfit([0, 1, 2], vals, 1)[0]
    df["dpm_slope3"] = df.apply(_slope, axis=1)

    df["team_name"] = df.groupby("nba_id")["team_name"].ffill()

    for h in [1, 2, 3, 4, 5]:
        df[f"target_dpm_{h}y"] = df.groupby("nba_id")["dpm"].shift(-h)

    df = df[df["season"] >= 2000]
    df = df.dropna(subset=["pts_36"]).reset_index(drop=True)
    return df


def merge_draft_scores(df: pd.DataFrame, draft_scores: pd.DataFrame | None) -> pd.DataFrame:
    """Join college draft-model scores by normalized name + draft year (optional)."""
    if draft_scores is None:
        df["draft_score"] = np.nan
        df["play_style_cluster"] = np.nan
        return df

    def norm(s):
        return str(s).lower().replace(".", "").replace("-", " ").strip()

    ds = draft_scores.copy()
    ds["name_key"] = ds["Name"].apply(norm)
    for col in ["draft_score", "play_style_cluster", "name_key"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["name_key"] = df["player_name"].apply(norm)
    df = df.merge(
        ds[["name_key", "draft_year", "draft_score", "play_style_cluster"]],
        on=["name_key", "draft_year"], how="left").drop(columns=["name_key"])
    return df


def add_team_context(df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Primary team per season, team win%, and teammate DPM."""
    logs = logs.copy()
    logs["season"] = logs["SEASON_YEAR"].astype(str).str[-2:].astype(int) + 2000

    mins = logs.groupby(["PLAYER_ID", "season", "TEAM_ABBREVIATION"])["MIN"].sum().reset_index()
    primary = mins.sort_values("MIN").groupby(["PLAYER_ID", "season"]).tail(1)
    primary = primary[["PLAYER_ID", "season", "TEAM_ABBREVIATION"]].rename(
        columns={"PLAYER_ID": "nba_id", "TEAM_ABBREVIATION": "team"})
    primary["nba_id"] = primary["nba_id"].astype(int)

    tg = logs[["season", "TEAM_ABBREVIATION", "GAME_ID", "WL"]].drop_duplicates()
    tg["win"] = (tg["WL"] == "W").astype(float)
    winpct = tg.groupby(["season", "TEAM_ABBREVIATION"])["win"].mean().reset_index()
    winpct = winpct.rename(columns={"TEAM_ABBREVIATION": "team", "win": "team_winpct"})

    df = df.merge(primary, on=["nba_id", "season"], how="left")
    team_dpm = df.groupby(["season", "team"])["dpm"].transform("sum")
    df["teammate_dpm"] = team_dpm - df["dpm"].fillna(0)
    df = df.merge(winpct, on=["season", "team"], how="left")
    df["teammate_dpm"] = df["teammate_dpm"].fillna(df["teammate_dpm"].median())
    df["team_winpct"] = df["team_winpct"].fillna(0.5)
    return df


def add_epm_features(df: pd.DataFrame, epm: pd.DataFrame,
                     actual: pd.DataFrame | None) -> pd.DataFrame:
    """Merge predictive EPM (now/lags/slope/teammate), actual EPM, and forward targets."""
    # idempotent: drop any prior EPM columns
    drop = [c for c in df.columns
            if c.startswith(("epm_", "oepm_", "depm_", "target_epm_",
                             "target_oepm_", "target_depm_"))
            or c in ("epm_now", "oepm_now", "depm_now", "teammate_epm", "usg_now")]
    df = df.drop(columns=drop, errors="ignore")

    df = df.merge(
        epm.rename(columns={"epm": "epm_now", "oepm": "oepm_now",
                            "depm": "depm_now", "p_usg": "usg_now"}),
        on=["nba_id", "season"], how="left")

    df = df.sort_values(["nba_id", "season"])
    for lag in [1, 2]:
        df[f"epm_lag{lag}"] = df.groupby("nba_id")["epm_now"].shift(lag)
        df[f"oepm_lag{lag}"] = df.groupby("nba_id")["oepm_now"].shift(lag)
        df[f"depm_lag{lag}"] = df.groupby("nba_id")["depm_now"].shift(lag)
    df["epm_delta1"] = df["epm_now"] - df["epm_lag1"]
    df["epm_delta2"] = df["epm_lag1"] - df["epm_lag2"]

    def _slope(row):
        vals = [row["epm_lag2"], row["epm_lag1"], row["epm_now"]]
        if any(pd.isna(v) for v in vals):
            return np.nan
        return np.polyfit([0, 1, 2], vals, 1)[0]
    df["epm_slope3"] = df.apply(_slope, axis=1)

    team_epm = df.groupby(["season", "team"])["epm_now"].transform("sum")
    df["teammate_epm"] = team_epm - df["epm_now"].fillna(0)
    df["teammate_epm"] = df["teammate_epm"].fillna(df["teammate_epm"].median())

    if actual is not None:
        df = df.merge(actual, on=["nba_id", "season"], how="left")
        df["epm_gap"] = df["epm_now"] - df["epm_actual_now"]
        df = df.sort_values(["nba_id", "season"])
        for lag in [1, 2]:
            df[f"epm_actual_lag{lag}"] = df.groupby("nba_id")["epm_actual_now"].shift(lag)
    else:
        for c in ["epm_actual_now", "oepm_actual", "depm_actual",
                  "epm_gap", "epm_actual_lag1", "epm_actual_lag2"]:
            df[c] = np.nan

    for h in [1, 2, 3, 4, 5]:
        fut = epm[["nba_id", "season", "epm", "oepm", "depm"]].copy()
        fut["season"] = fut["season"] - h
        fut = fut.rename(columns={"epm": f"target_epm_{h}y",
                                  "oepm": f"target_oepm_{h}y",
                                  "depm": f"target_depm_{h}y"})
        df = df.merge(fut, on=["nba_id", "season"], how="left")
    return df


def apply_survivorship(df: pd.DataFrame) -> pd.DataFrame:
    """True washouts (career ended before season+h) get a replacement-level target
    instead of being dropped, so the model isn't biased toward survivors.
    Idempotent via _raw_* backups."""
    cur = df["season"].max()
    player_last = df.groupby("nba_id")["season"].transform("max")
    for prefix, nowcol in [("target_epm", "epm_now"), ("target_dpm", "dpm")]:
        dnp = df[nowcol].quantile(0.05)
        for h in [1, 2, 3, 4, 5]:
            tgt, raw = f"{prefix}_{h}y", f"_raw_{prefix}_{h}y"
            if raw in df.columns:
                df[tgt] = df[raw]
            else:
                df[raw] = df[tgt].copy()
            fill = ((df["season"] + h) <= cur) & df[tgt].isna() & (player_last < (df["season"] + h))
            df.loc[fill, tgt] = dnp
    return df


def add_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Availability/role context. Interactions clip impact at 0 so they only ever
    *reward* good-and-played-a-lot, never penalize negative high-minute players."""
    df = df.copy()
    # COVID-shortened seasons were not 82 games; boost their games to an 82-game
    # equivalent so availability features are comparable across seasons (a healthy
    # 2021 player reads as fully available, an injured one stays proportionally low).
    SEASON_GAMES = {2020: 73, 2021: 72}
    season_len = df["season"].map(SEASON_GAMES).fillna(82)
    games_82 = df["games"] * (82 / season_len)

    df["games_played_pct"] = games_82 / 82
    df["total_minutes"] = games_82 * df["min_pg"]
    df["minutes_x_games"] = df["min_pg"] * df["games_played_pct"]

    df["epm_x_games"] = df["epm_now"].clip(lower=0) * df["games_played_pct"]
    df["epm_x_min_pg"] = df["epm_now"].clip(lower=0) * df["min_pg"]
    df["dpm_x_games"] = df["dpm"].clip(lower=0) * df["games_played_pct"]
    df["dpm_x_min_pg"] = df["dpm"].clip(lower=0) * df["min_pg"]
    df["epm_actual_x_games"] = df["epm_actual_now"].clip(lower=0) * df["games_played_pct"]
    df["epm_actual_x_min_pg"] = df["epm_actual_now"].clip(lower=0) * df["min_pg"]

    # low-games flags use the boosted count so short seasons don't over-flag healthy players
    df["low_games_high_minutes"] = ((games_82 < 50) & (df["min_pg"] >= 24)).astype(int)
    df["low_games_low_minutes"] = ((games_82 < 50) & (df["min_pg"] < 15)).astype(int)
    df["low_games_good_epm"] = ((games_82 < 50) & (df["epm_now"] > 0)).astype(int)
    df["low_games_bad_epm"] = ((games_82 < 50) & (df["epm_now"] < -1)).astype(int)
    df["low_games_good_dpm"] = ((games_82 < 50) & (df["dpm"] > 0)).astype(int)
    df["low_games_bad_dpm"] = ((games_82 < 50) & (df["dpm"] < -1)).astype(int)
    return df


def add_young_improver(df: pd.DataFrame) -> pd.DataFrame:
    """Flag young, fast-rising players (EPM and DPM variants)."""
    df = df.copy()
    df["epm_change_1y"] = df["epm_now"] - df["epm_lag1"]
    df["epm_change_2y"] = df["epm_now"] - df["epm_lag2"]
    df["young_improver"] = (
        (df["years_in_league"] <= 3) & (df["age"] <= 23) &
        ((df["epm_change_1y"] >= 1.0) | (df["epm_change_2y"] >= 2.0))
    ).astype(int)
    df["dpm_change_1y"] = df["dpm"] - df["dpm_lag1"]
    df["dpm_change_2y"] = df["dpm"] - df["dpm_lag2"]
    df["young_improver_dpm"] = (
        (df["years_in_league"] <= 3) & (df["age"] <= 23) &
        ((df["dpm_change_1y"] >= 1.0) | (df["dpm_change_2y"] >= 2.0))
    ).astype(int)
    return df


def build_training_table(darko, draft, logs, epm, actual, draft_scores) -> pd.DataFrame:
    """Run the full feature pipeline end to end."""
    df = build_base_dataset(darko, draft, logs)
    df = merge_draft_scores(df, draft_scores)
    df = add_team_context(df, logs)
    df = add_epm_features(df, epm, actual)
    df = apply_survivorship(df)
    df = add_availability_features(df)
    df = add_young_improver(df)
    return df
