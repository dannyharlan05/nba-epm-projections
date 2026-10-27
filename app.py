"""Streamlit front end for the EPM projection models.

Loads pre-built artifacts (the notebook's export / build_artifacts.py), so it never
retrains on launch. A sidebar toggle switches the projection target between the
predictive (stabilized) EPM model and the observed (raw-season) EPM model.

    streamlit run app.py
"""
from __future__ import annotations
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.isotonic import IsotonicRegression

ART = "artifacts"
HORIZONS = [1, 2, 3, 4, 5]
# "If the season ended today" view (artifacts/predictions_now.parquet). Off until
# Dunks & Threes posts observed EPM for the season in progress.
SHOW_SEASON_SO_FAR = False

st.set_page_config(page_title="NBA EPM Projections", layout="wide")


@st.cache_data
def load_artifacts():
    df = pd.read_parquet(os.path.join(ART, "predictions.parquet"))
    with open(os.path.join(ART, "meta.json")) as f:
        meta = json.load(f)
    return df, meta


@st.cache_data
def load_now():
    """This season so far, run through the models (notebook's season-ended-today cell)."""
    p = os.path.join(ART, "predictions_now.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


@st.cache_data
def load_wins():
    """Hand-entered preseason win projections (artifacts/win_projections.csv)."""
    p = os.path.join(ART, "win_projections.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_current_teams():
    """Current NBA rosters (artifacts/current_teams.csv, written by the notebook's roster cell)."""
    p = os.path.join(ART, "current_teams.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _name_key(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[`'.,]", "", s).replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def apply_current_teams(cur: pd.DataFrame) -> pd.DataFrame:
    """Replace last season's team with the player's team now; not on a roster -> FA.
    Matches on nba_id when the artifacts carry it, otherwise on the player's name."""
    ct = load_current_teams()
    if ct is None:
        return cur
    cur = cur.copy()
    if "nba_id" in cur.columns:
        team = cur["nba_id"].map(ct.drop_duplicates("nba_id").set_index("nba_id")["team"])
    else:
        names = [ct[["player_name", "team"]]]
        if "data_name" in ct.columns:                    # spelling used in the data, if different
            names.append(ct.dropna(subset=["data_name"])[["data_name", "team"]]
                         .rename(columns={"data_name": "player_name"}))
        by_name = (pd.concat(names).assign(k=lambda x: x["player_name"].map(_name_key))
                   .drop_duplicates(["k", "team"]).drop_duplicates("k", keep=False)
                   .set_index("k")["team"])
        team = cur["player_name"].map(_name_key).map(by_name)
    cur["team"] = team.fillna("FA")
    return cur


@st.cache_data
def load_players():
    """Rookie season + draft info per player (artifacts/players.csv, NBA API all-players + draft history)."""
    p = os.path.join(ART, "players.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def apply_player_info(cur: pd.DataFrame) -> pd.DataFrame:
    """Add rookie_season, draft_year, draft_round, draft_pick. Matches on nba_id when the
    artifacts carry it; otherwise via the roster file's nba_id, then on the player's name
    (a name shared by two players goes to the more recent one)."""
    pl = load_players()
    if pl is None:
        return cur
    cur = cur.copy()
    info = pl.drop_duplicates("nba_id").set_index("nba_id")
    nid = cur["nba_id"] if "nba_id" in cur.columns else pd.Series(np.nan, index=cur.index)
    ct = load_current_teams()
    if ct is not None and "data_name" in ct.columns:
        by_roster = (ct.dropna(subset=["data_name"]).assign(k=lambda x: x["data_name"].map(_name_key))
                     .drop_duplicates("k", keep=False).set_index("k")["nba_id"])
        nid = nid.fillna(cur["player_name"].map(_name_key).map(by_roster))
    # by name: of the players with that name, the one whose career covers the row's season
    m = (pd.DataFrame({"k": cur["player_name"].map(_name_key), "season": cur["season"]}).reset_index()
         .merge(pl.assign(k=pl["player_name"].map(_name_key))[["k", "nba_id", "rookie_season", "last_season"]],
                on="k"))
    m = m[(m["season"] >= m["rookie_season"] - 1) & (m["season"] <= m["last_season"] + 1)]
    m = m.sort_values("last_season").drop_duplicates("index", keep="last").set_index("index")["nba_id"]
    nid = nid.fillna(m.reindex(cur.index))
    for c in ["rookie_season", "draft_year", "draft_round", "draft_pick"]:
        cur[c] = nid.map(info[c])
    return cur


if not os.path.exists(os.path.join(ART, "predictions.parquet")):
    st.error("No artifacts found. Run `python build_artifacts.py` first.")
    st.stop()

df, meta = load_artifacts()
CUR = meta["current_season"]

# players who sat out the whole season carry status != "played"; older artifacts
# have no such column, so default everyone to played
if "status" not in df.columns:
    df["status"] = "played"


def _peak_smooth(v):
    """Best unimodal fit to one path [now, 1y..5y]: may rise to a peak then fall, never
    falls then rises. Year 0 is the observed value, so it is pinned (huge weight) and only
    the projections move. Same fit as top.ipynb (FORM='peak', ANCHOR_ACTUAL=True)."""
    yr = np.arange(len(v), dtype=float)
    w = np.ones(len(v))
    w[0] = 1e6
    best, best_sse = v, np.inf
    for k in range(len(yr)):
        up = IsotonicRegression(increasing=True).fit_transform(yr[:k + 1], v[:k + 1], sample_weight=w[:k + 1])
        dn = IsotonicRegression(increasing=False).fit_transform(yr[k:], v[k:], sample_weight=w[k:])
        cand = np.concatenate([up[:-1], dn])
        cand[:k] = np.minimum(cand[:k], cand[k])
        sse = (w * (v - cand) ** 2).sum()
        if sse < best_sse:
            best, best_sse = cand, sse
    # points pooled with the pinned year 0 land ~1e-7 off it; snap them so flat is flat
    best[np.abs(best - v[0]) < 1e-5] = v[0]
    return best


@st.cache_data
def smooth_paths(cur: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the current season's projections with their smoothed paths, for both
    targets. Rows missing the current value or any horizon are left as-is."""
    cur = cur.copy()
    for now_col, pp in [("epm_now", "pred_epm"), ("epm_actual_now", "pred_epm_actual")]:
        cols = [now_col] + [f"{pp}_{h}y" for h in HORIZONS]
        if not set(cols) <= set(cur.columns):
            continue
        ok = cur[cols].notna().all(axis=1)
        if ok.any():
            sm = np.apply_along_axis(_peak_smooth, 1, cur.loc[ok, cols].to_numpy(float))
            cur.loc[ok, cols[1:]] = sm[:, 1:]
    return cur


def season_label(s):
    return f"{int(s) - 1}-{str(int(s))[-2:]}"

has_actual = "pred_epm_actual_1y" in df.columns and "cv_mae_actual" in meta

# ----- view: projections made going into this season, or from this season so far -----
df_now = load_now()
VIEW_INTO = f"Going into {season_label(CUR + 1)}"
views = [VIEW_INTO]
if SHOW_SEASON_SO_FAR and df_now is not None and len(df_now):
    NOW_SEASON = int(df_now["season"].max())
    views.append(f"If {season_label(NOW_SEASON)} ended today")

st.title("NBA EPM Projections")
view = st.radio("View", views, horizontal=True,
                help="Going into the season: projections from last season's full numbers. "
                     "Ended today: the same models run on this season's numbers so far, "
                     "as if the season stopped now.") if len(views) > 1 else VIEW_INTO
IS_NOW = view != VIEW_INTO
if IS_NOW:
    df, CUR = df_now, NOW_SEASON
    if "status" not in df.columns:
        df["status"] = "played"
    has_actual = False          # observed EPM isn't used for the season in progress
    _asof = df["as_of"].max() if "as_of" in df.columns else pd.NaT
    st.caption(f"{season_label(CUR)} so far"
               + (f" · regular-season games through {pd.Timestamp(_asof):%b %d}" if pd.notna(_asof)
                  else " · no regular-season games yet (box stats are EPM's own estimates)")
               + " · predictive EPM only · early-season numbers swing a lot")
else:
    st.caption(f"Multi-year Estimated Plus-Minus forecasts · current season {CUR} · "
               "time-ordered cross-validation")
NOW_LABEL = "so far" if IS_NOW else "current"

# ----- projection target toggle (click to switch model; drives every tab) -----
if has_actual:
    target = st.radio(
        "Projection target",
        ["Predictive EPM (stabilized)", "Observed EPM (raw season)"],
        horizontal=True,
        help="Predictive EPM is the smoothed, next-day number (easier to project). "
             "Observed EPM is the raw full-season result (noisier, harder to project).",
    )
else:
    target = "Predictive EPM (stabilized)"
IS_PRED = target.startswith("Predictive")
NOW = "epm_now" if IS_PRED else "epm_actual_now"
PP = "pred_epm" if IS_PRED else "pred_epm_actual"
MAE_BY_H = {int(r["horizon"]): float(r["cv_mae"]) for r in
            (meta["cv_mae"] if IS_PRED else meta["cv_mae_actual"])}


def pcolname(h):
    return f"{PP}_{h}y"


# fixed y-axis range so every player's chart is on the same scale
_pred_cols = [pcolname(h) for h in HORIZONS]
_vals = pd.concat([df[c] for c in _pred_cols] + [df[NOW]]).dropna()
Y_RANGE = [float(_vals.min()) - 0.7, float(_vals.max()) + 0.7]

# current-season slice + league rank by current value
current = apply_player_info(apply_current_teams(smooth_paths(df[df["season"] == CUR])))
if "rookie_season" in current.columns:
    current["years_in_league"] = CUR - current["rookie_season"] + 1     # 1 = rookie in season CUR
current["league_rank"] = current[NOW].rank(ascending=False, method="min")
NAMES = current.dropna(subset=[NOW]).sort_values(NOW, ascending=False)["player_name"].dropna().unique().tolist()


def filter_players(b: pd.DataFrame, key: str, season: int) -> pd.DataFrame:
    """Team / age / years in league / draft filters (collapsed). `season` is the season
    the rows describe, so years in league counts up to it."""
    with st.expander("Filters"):
        f1, f2, f3 = st.columns(3)
        teams = sorted(t for t in b["team"].dropna().unique() if t != "FA")
        teams += ["FA"] if (b["team"] == "FA").any() else []
        pick_teams = f1.multiselect("Team", teams, placeholder="All teams", key=f"{key}_team")
        a_lo, a_hi = int(np.floor(b["age"].min())), int(np.ceil(b["age"].max()))
        ages = f2.slider("Age", a_lo, a_hi, (a_lo, a_hi), key=f"{key}_age")
        has_dnp = b["status"].ne("played").any()
        hide_dnp = f3.checkbox(f"Hide players who missed {season_label(season)}",
                               key=f"{key}_dnp") if has_dnp else False
        if "rookie_season" in b.columns:
            b = b.assign(years_in_league=season - b["rookie_season"] + 1)
            g1, g2, g3 = st.columns(3)
            y_hi = max(int(b["years_in_league"].fillna(1).max()), 2)
            yrs_in = g1.slider(f"Years in league (1 = rookie in {season_label(season)})",
                               1, y_hi, (1, y_hi), key=f"{key}_yrs")
            classes = sorted(b["draft_year"].dropna().astype(int).unique(), reverse=True)
            pick_classes = g2.multiselect("Draft class", classes + ["Undrafted"],
                                          placeholder="All classes", key=f"{key}_class")
            rnd = g3.radio("Draft round", ["Any", "1st", "2nd", "Undrafted"], horizontal=True,
                           key=f"{key}_round")

    if pick_teams:
        b = b[b["team"].isin(pick_teams)]
    b = b[b["age"].between(*ages) | b["age"].isna()]
    if hide_dnp:
        b = b[b["status"].eq("played")]
    if "rookie_season" in b.columns:
        undrafted = b["draft_year"].isna() & b["rookie_season"].notna()
        if yrs_in != (1, y_hi):
            keep = b["years_in_league"].between(*yrs_in)
            b, undrafted = b[keep], undrafted[keep]
        if pick_classes:
            keep = b["draft_year"].isin([c for c in pick_classes if c != "Undrafted"])
            if "Undrafted" in pick_classes:
                keep |= undrafted
            b, undrafted = b[keep], undrafted[keep]
        if rnd == "Undrafted":
            b = b[undrafted]
        elif rnd != "Any":
            b = b[b["draft_round"].eq({"1st": 1, "2nd": 2}[rnd])]
    return b


def proj_values(row):
    """[current, +1y ... +5y] and the matching seasons."""
    yrs = [CUR] + [CUR + h for h in HORIZONS]
    vals = [row.get(NOW)] + [row.get(pcolname(h)) for h in HORIZONS]
    return yrs, vals


wins = load_wins()
_names = (["Leaderboards", "Player", "Compare", "Backtest"]
          + (["Win projections"] if wins is not None else []) + ["Methodology"])
_tabs = dict(zip(_names, st.tabs(_names)))
tab_board, tab_player, tab_compare = _tabs["Leaderboards"], _tabs["Player"], _tabs["Compare"]
tab_back, tab_method = _tabs["Backtest"], _tabs["Methodology"]
tab_wins = _tabs.get("Win projections")

# ============================================================ Player
with tab_player:
    default = NAMES.index("Nikola Jokic") if "Nikola Jokic" in NAMES else 0
    name = st.selectbox("Player", NAMES, index=default)
    row = current[current["player_name"] == name].iloc[0]

    age = f"Age {row['age']:.0f}" if pd.notna(row.get("age")) else ""
    st.markdown(f"### {name} · {age}")

    if row.get("status", "played") != "played":
        st.warning(
            f"Did not play in {season_label(CUR)} — every projection below assumes he "
            + ("returns. Predictive EPM is still published for him, so the current value "
               "is a live estimate, not a result."
               if IS_PRED else
               f"returns. He has no {season_label(CUR)} result, so the current value shown "
               f"is his {season_label(CUR - 1)} season.")
        )

    now_val = row[NOW]

    cols = st.columns(4)
    cols[0].metric(f"EPM {NOW_LABEL}", f"{now_val:+.2f}" if pd.notna(now_val) else "n/a")
    for i, h in enumerate([1, 3, 5]):
        p = row.get(pcolname(h))
        cols[i + 1].metric(f"{h}-Year ({CUR + h})", f"{p:+.2f}" if pd.notna(p) else "n/a")

    rcols = st.columns(2)
    rank = row.get("league_rank")
    rcols[0].metric("Current EPM rank", f"#{int(rank)}" if pd.notna(rank) else "n/a")
    p5 = row.get(pcolname(5))
    five = (p5 - now_val) if (pd.notna(p5) and pd.notna(now_val)) else None
    rcols[1].metric("5-year change", f"{five:+.2f}" if five is not None else "n/a")

    yrs, vals = proj_values(row)
    mae = [0.0] + [MAE_BY_H.get(h, 0.0) for h in HORIZONS]
    upper = [v + m if pd.notna(v) else None for v, m in zip(vals, mae)]
    lower = [v - m if pd.notna(v) else None for v, m in zip(vals, mae)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=upper, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yrs, y=lower, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
                             name="typical out-of-fold error", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yrs, y=vals, mode="lines+markers",
                             line=dict(color="#1f77b4", width=3), marker=dict(size=7),
                             name="EPM path", hovertemplate="%{x}: %{y:+.2f}<extra></extra>"))
    if pd.notna(now_val):
        fig.add_trace(go.Scatter(x=[CUR], y=[now_val], mode="markers",
                                 marker=dict(size=13, color="#111"),
                                 name=f"{NOW_LABEL} (actual)",
                                 hovertemplate=NOW_LABEL + " %{y:+.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Season (end year)", yaxis_title="EPM",
                      height=440, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(range=Y_RANGE)
    fig.update_xaxes(tickmode="array", tickvals=yrs)
    st.plotly_chart(fig, width="stretch")
    st.caption("Solid dot = current (actual) EPM. Shaded band = typical out-of-fold "
               "absolute error at each horizon (not a confidence interval). "
               "Projected paths are smoothed so they never dip and then recover; "
               "see Methodology.")

# ============================================================ Compare
with tab_compare:
    picks = st.multiselect("Players (2–4)", NAMES,
                           default=[n for n in ["Nikola Jokic", "Victor Wembanyama"] if n in NAMES],
                           max_selections=4)
    if len(picks) < 2:
        st.info("Pick at least two players to compare.")
    else:
        fig = go.Figure()
        rows = []
        for nm in picks:
            r = current[current["player_name"] == nm].iloc[0]
            yrs, vals = proj_values(r)
            fig.add_trace(go.Scatter(x=yrs, y=vals, mode="lines+markers", name=nm,
                                     hovertemplate=nm + " %{x}: %{y:+.2f}<extra></extra>"))
            rows.append({
                "Player": nm,
                "Age": r.get("age"),
                "EPM now": r.get(NOW),
                "1-Year": r.get(pcolname(1)),
                "3-Year": r.get(pcolname(3)),
                "5-Year": r.get(pcolname(5)),
            })
        fig.update_layout(xaxis_title="Season (end year)", yaxis_title="EPM",
                          height=460, hovermode="x unified")
        fig.update_yaxes(range=Y_RANGE)
        fig.update_xaxes(tickmode="array", tickvals=[CUR] + [CUR + h for h in HORIZONS])
        st.plotly_chart(fig, width="stretch")

        comp = pd.DataFrame(rows)
        st.dataframe(
            comp.style.format({"Age": "{:.0f}", "EPM now": "{:+.2f}", "1-Year": "{:+.2f}",
                               "3-Year": "{:+.2f}", "5-Year": "{:+.2f}"}),
            hide_index=True, width="stretch")

# ============================================================ Leaderboards
with tab_board:
    c1, c2 = st.columns(2)
    h = c1.selectbox("Years ahead", HORIZONS, index=2, key="lb_h")
    search = c2.text_input("Search player")

    pcol = pcolname(h)
    b = current.dropna(subset=[pcol, NOW]).copy()
    if search:
        b = b[b["player_name"].str.contains(search, case=False, na=False)]

    b = filter_players(b, "lb", CUR)
    b["change"] = (b[pcol] - b[NOW]).astype(float)
    b.loc[b[pcol] < -2, "change"] = float("nan")
    b = b.sort_values(pcol, ascending=False, na_position="last").reset_index(drop=True)
    b.insert(0, "Rank", b.index + 1)
    b["_player"] = b["player_name"].where(
        b["status"].eq("played"), b["player_name"] + "  · DNP")
    info = []
    if "years_in_league" in b.columns:
        b["_draft"] = np.where(b["draft_year"].notna(),
                               b["draft_year"].astype("Int64").astype(str) + " #" + b["draft_pick"].astype("Int64").astype(str),
                               np.where(b["rookie_season"].notna(), "Undrafted", None))
        info = ["years_in_league", "_draft"]
    out = b[["Rank", "_player", "team", "age"] + info + [NOW, pcol, "change"]].copy()
    out.columns = (["Rank", "Player", "Team", "Age"] + (["Yrs", "Draft"] if info else [])
                   + ["EPM now", f"Proj {h}y", "Change"])

    n_dnp = int(b["status"].ne("played").sum())
    st.caption(f"{len(out)} players · projections smoothed across horizons (see Methodology)"
               " · blank Change = projected below -2"
               + (f" · DNP = did not play in {season_label(CUR)} ({n_dnp} players), "
                  f"projection assumes a return" if n_dnp else ""))
    st.dataframe(
        out, hide_index=True, width="stretch", height=640,
        column_config={
            "Age": st.column_config.NumberColumn(format="%d"),
            "Yrs": st.column_config.NumberColumn(format="%d", help=f"Seasons in the league through {season_label(CUR)}"),
            "EPM now": st.column_config.NumberColumn(format="%+.2f"),
            f"Proj {h}y": st.column_config.NumberColumn(format="%+.2f"),
            "Change": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
    csv = out.copy()
    csv["Change"] = csv["Change"].map(lambda v: "DNQ" if pd.isna(v) else f"{v:+.2f}")
    st.download_button("Download CSV", csv.to_csv(index=False).encode(),
                       file_name=f"epm_{'predictive' if IS_PRED else 'observed'}_{h}y.csv",
                       mime="text/csv")

# ============================================================ Backtest
def _birth(d):
    """Approximate birth year: tells apart two players who share a name."""
    return (d["season"] - d["age"]).round()


def with_actual(past: pd.DataFrame, h: int) -> pd.Series:
    """What each player's EPM actually was h seasons later (NaN = didn't play that season)."""
    later = df[(df["season"] == past["season"].iloc[0] + h) & df["status"].eq("played")]
    later = later[["player_name", "season", "age", NOW]]
    later = later.assign(b=_birth(later)).drop_duplicates(["player_name", "b"])
    key = past.assign(b=_birth(past))[["player_name", "b"]].reset_index()
    m = key.merge(later[["player_name", "b", NOW]], on="player_name", how="left")
    m = m[(m["b_x"] - m["b_y"]).abs() <= 1].drop_duplicates("index")
    return m.set_index("index")[NOW].reindex(past.index)


_cov = df.groupby("season")[[pcolname(h) for h in HORIZONS]].apply(lambda x: x.notna().all(axis=1).mean())
BT_SEASONS = [int(x) for x in _cov[(_cov > 0.9) & (_cov.index < CUR)].index]

with tab_back:
    if IS_NOW or not BT_SEASONS:
        st.info("The backtest uses the going-into-the-season view.")
    else:
        st.caption("What the model would have said going into each past season. It was built "
                   "afterwards, so this is a backtest, not a record of real-time calls: each season's "
                   "numbers come from a model trained only on earlier seasons (out-of-fold), "
                   "smoothed the same way as the current projections. Model design and "
                   "settings were still chosen with hindsight.")
        mode = st.radio("Show", ["Season", "Player history"], horizontal=True, key="bt_mode")

        if mode == "Season":
            c1, c2, c3 = st.columns(3)
            S = c1.selectbox("Going into", BT_SEASONS[::-1], format_func=lambda x: season_label(x + 1),
                             key="bt_s")
            hs = [x for x in HORIZONS if S + x <= CUR]
            bh = c2.selectbox("Years ahead", hs, index=0, key="bt_h",
                              format_func=lambda x: f"{x} ({season_label(S + x)})")
            bsearch = c3.text_input("Search player", key="bt_search")

            past = apply_player_info(smooth_paths(df[df["season"] == S]))
            past = past.dropna(subset=[NOW, pcolname(bh)]).copy()
            past["actual"] = with_actual(past, bh)
            past["miss"] = past["actual"] - past[pcolname(bh)]
            past = filter_players(past, "bt", S)
            ok = past["actual"].notna()
            mae, base = past.loc[ok, "miss"].abs().mean(), (past.loc[ok, "actual"] - past.loc[ok, NOW]).abs().mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Players with a result", f"{int(ok.sum())} / {len(past)}")
            m2.metric("Model MAE", f"{mae:.2f}" if ok.any() else "n/a")
            m3.metric("No-change MAE", f"{base:.2f}" if ok.any() else "n/a", help=f"Guessing every player stays at his {season_label(S)} EPM")
            m4.metric("Within 1 point", f"{(past.loc[ok, 'miss'].abs() <= 1).mean():.0%}" if ok.any() else "n/a")

            if bsearch:
                past = past[past["player_name"].str.contains(bsearch, case=False, na=False)]
            past = past.sort_values(pcolname(bh), ascending=False).reset_index(drop=True)
            past.insert(0, "Rank", past.index + 1)
            past["_draft"] = np.where(past["draft_year"].notna(),
                                      past["draft_year"].astype("Int64").astype(str) + " #"
                                      + past["draft_pick"].astype("Int64").astype(str),
                                      np.where(past["rookie_season"].notna(), "Undrafted", None))
            bo = past[["Rank", "player_name", "team", "age", "years_in_league", "_draft",
                       NOW, pcolname(bh), "actual", "miss"]]
            bo.columns = ["Rank", "Player", f"Team ({season_label(S)})", "Age", "Yrs", "Draft", f"EPM {season_label(S)}",
                          f"Proj {season_label(S + bh)}", f"Actual {season_label(S + bh)}", "Miss"]
            st.caption("Ranked by projection · Miss = actual minus projection (positive = beat it) · "
                       "blank Actual = didn't play that season"
                       + (" · 2+ years ahead reads slightly optimistic: the cross-validation let "
                          "training seasons' later targets overlap the test years" if bh > 1 else ""))
            fmt = {c: st.column_config.NumberColumn(format="%+.2f") for c in bo.columns[6:]}
            st.dataframe(bo, hide_index=True, width="stretch", height=640,
                         column_config={"Age": st.column_config.NumberColumn(format="%d"),
                                        "Yrs": st.column_config.NumberColumn(format="%d"), **fmt})


        else:
            hist_names = (df[df["season"].isin(BT_SEASONS)].sort_values(NOW, ascending=False)
                          ["player_name"].dropna().unique().tolist())
            default = hist_names.index("LeBron James") if "LeBron James" in hist_names else 0
            nm = st.selectbox("Player", hist_names, index=default, key="bt_player")
            ph = df[df["player_name"] == nm].assign(b=lambda d: _birth(d))
            ph = ph[(ph["b"] - ph["b"].iloc[-1]).abs() <= 1].sort_values("season")   # same person
            fig = go.Figure()
            for _, r in smooth_paths(ph[ph["season"].isin(BT_SEASONS)]).iterrows():
                yrs = [r["season"]] + [r["season"] + x for x in HORIZONS]
                fig.add_trace(go.Scatter(
                    x=yrs, y=[r[NOW]] + [r[pcolname(x)] for x in HORIZONS], mode="lines",
                    line=dict(width=1.5, dash="dot"), opacity=0.55, showlegend=False,
                    name=f"going into {season_label(r['season'] + 1)}",
                    hovertemplate="%{x}: %{y:+.2f}<extra>going into " + season_label(r["season"] + 1) + "</extra>"))
            fig.add_trace(go.Scatter(x=ph["season"], y=ph[NOW], mode="lines+markers",
                                     line=dict(color="#111", width=3), marker=dict(size=7), name="actual",
                                     hovertemplate="%{x}: %{y:+.2f}<extra>actual</extra>"))
            fig.update_layout(xaxis_title="Season (end year)", yaxis_title="EPM", height=480,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig.update_yaxes(range=Y_RANGE)
            st.plotly_chart(fig, width="stretch")
            st.caption("Black = what actually happened. Each dotted line is the 5-year path the model "
                       "would have projected going into that season. Backtest only: out-of-fold "
                       "predictions from models trained on earlier seasons.")

# ============================================================ Win projections
if tab_wins is not None:
    with tab_wins:
        st.caption(f"Preseason win projections for {season_label(wins['season'].max())} · "
                   f"{wins['wins'].sum():g} total wins · based on the 1-year observed-EPM projections, "
                   "made before the EPM methodology update")
        for col, conf in zip(st.columns(2), ["West", "East"]):
            t = wins[wins["conf"] == conf].sort_values("wins", ascending=False).reset_index(drop=True)
            t.insert(0, "Rank", t.index + 1)
            t["Record"] = t["wins"].map(lambda w: f"{w:g}-{82 - w:g}")
            col.subheader(conf)
            col.dataframe(
                t[["Rank", "team", "wins", "Record"]].rename(columns={"team": "Team", "wins": "Wins"}),
                hide_index=True, width="stretch", height=38 + 35 * len(t),
                column_config={"Wins": st.column_config.ProgressColumn(
                    format="%.1f", min_value=0, max_value=82)})

# ============================================================ Methodology
with tab_method:
    pmae = ", ".join(f"{r['cv_mae']:.2f}" for r in meta["cv_mae"])
    amae = ", ".join(f"{r['cv_mae']:.2f}" for r in meta["cv_mae_actual"]) if has_actual else "n/a"
    NOW_SECTION = ("""
### If the season ended today

The second view runs the same trained models on the season in progress, as if it stopped
now: this season's predicted EPM and DARKO so far, regular-season box stats so far (players
with no qualifying games yet use EPM's own per-100 estimates), and games played scaled to
a full season. Nothing is retrained on the partial season. It is predictive EPM only, and
early in the year it rests on a handful of games, so expect it to move a lot week to week.

""" if len(views) > 1 else "")
    st.markdown(f"""
### What this is

A model that projects each NBA player's Estimated Plus-Minus (EPM) one to five
seasons into the future. You can switch the **projection target** in the sidebar:

- **Predictive EPM (stabilized)** — the smoothed, next-day impact estimate. Already
  noise-reduced, so it's the easier, better-behaved thing to project.
  Out-of-fold MAE by horizon (1–5y): **{pmae}**.
- **Observed EPM (raw season)** — the actual full-season result. It swings with
  injuries, hot/cold stretches, and role changes, so it's genuinely harder to predict.
  Out-of-fold MAE by horizon: **{amae}**.

The MAE gap between the two is the point: a stabilized metric exists precisely because
the raw season number is noisy, and that noise shows up as a higher, irreducible error
on the observed-EPM model.

### How it works

- **Features:** current and lagged EPM, DARKO DPM as a second impact signal, observed
  EPM, per-36 box-score rates, team context, and minutes/impact
  interactions. Both models share the same feature set.
- **Time-ordered validation:** every cross-validation fold trains only on seasons before
  the test seasons, so no future information reaches the feature set. Reported numbers are
  out-of-fold.
- **Survivorship handling:** players who leave the league are not silently dropped,
  which would bias the model toward survivors. Their future is decayed from their last
  level toward replacement over the horizon (a gradual fade, not an instant cliff), so
  decline is modeled rather than ignored. The same handling is applied to both targets.

{NOW_SECTION}### Smoothed trajectories

Each horizon (1y through 5y) is its own model, so left alone a player's path can go
down and then come back up purely from model-to-model noise. Before display, every
current path (current EPM → 1y … 5y) is replaced by its closest **single-peaked** fit:
it can rise to a peak and then decline, but it never declines and then recovers. Current
EPM is a real result, so it is never changed; only the projections move.

This is a presentation choice, **not an accuracy gain**. Backtested on the out-of-fold
history it changes roughly a third (predictive) to two-fifths (observed) of paths but moves MAE by less than 0.005 at every
horizon, in either direction. The MAE figures above are for the unsmoothed model output.
A genuine "regress next year, then grow" path for a young player gets flattened.

### Limitations

- Gradient-boosted trees cannot extrapolate beyond the range seen in training, so an
  unprecedented young season is pulled toward historical precedent and the projection
  will read conservative for true outliers.
- EPM inputs begin in the early 2000s, so seasons before then are not covered.
""")
