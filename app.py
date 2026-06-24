"""Streamlit front end for the EPM projection model.

Loads pre-built artifacts (the notebook's export / build_artifacts.py), so it never
retrains on launch. Tabs: player profile, compare, leaderboards, methodology.

    streamlit run app.py
"""
from __future__ import annotations
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = "artifacts"
HORIZONS = [1, 2, 3, 4, 5]

st.set_page_config(page_title="NBA EPM Projections", layout="wide")


@st.cache_data
def load_artifacts():
    df = pd.read_parquet(os.path.join(ART, "predictions.parquet"))
    with open(os.path.join(ART, "meta.json")) as f:
        meta = json.load(f)
    return df, meta


if not os.path.exists(os.path.join(ART, "predictions.parquet")):
    st.error("No artifacts found. Run `python build_artifacts.py` first.")
    st.stop()

df, meta = load_artifacts()
CUR = meta["current_season"]

# per-horizon out-of-fold MAE (for the error band)
MAE_BY_H = {int(r["horizon"]): float(r["cv_mae"]) for r in meta["cv_mae"]}

# fixed y-axis range so every player's chart is on the same scale
_pred_cols = [f"pred_epm_{h}y" for h in HORIZONS]
_vals = pd.concat([df[c] for c in _pred_cols] + [df["epm_now"]]).dropna()
Y_RANGE = [float(_vals.min()) - 0.7, float(_vals.max()) + 0.7]

# current-season slice + league rank by current EPM
current = df[df["season"] == CUR].copy()
current["league_rank"] = current["epm_now"].rank(ascending=False, method="min")
NAMES = current.sort_values("epm_now", ascending=False)["player_name"].dropna().unique().tolist()


def proj_values(row):
    """[current, +1y ... +5y] and the matching seasons."""
    yrs = [CUR] + [CUR + h for h in HORIZONS]
    vals = [row.get("epm_now")] + [row.get(f"pred_epm_{h}y") for h in HORIZONS]
    return yrs, vals


st.title("NBA EPM Projections")
st.caption(f"Multi-year Estimated Plus-Minus forecasts · current season {CUR} · "
           "leak-free temporal cross-validation")

tab_board, tab_player, tab_compare, tab_method = st.tabs(
    ["Leaderboards", "Player", "Compare", "Methodology"])

# ============================================================ Player
with tab_player:
    default = NAMES.index("Nikola Jokic") if "Nikola Jokic" in NAMES else 0
    name = st.selectbox("Player", NAMES, index=default)
    row = current[current["player_name"] == name].iloc[0]

    age = f"Age {row['age']:.0f}" if pd.notna(row.get("age")) else ""
    st.markdown(f"### {name} · {age}")

    epm_now = row["epm_now"]

    # headline metric row: current + each horizon (values only, no arrows)
    cols = st.columns(4)
    cols[0].metric("Current EPM", f"{epm_now:+.2f}" if pd.notna(epm_now) else "n/a")
    for i, h in enumerate([1, 3, 5]):
        p = row.get(f"pred_epm_{h}y")
        cols[i + 1].metric(f"{h}-Year ({CUR + h})", f"{p:+.2f}" if pd.notna(p) else "n/a")

    # insight row
    rcols = st.columns(2)
    rank = row.get("league_rank")
    rcols[0].metric("Current EPM rank", f"#{int(rank)}" if pd.notna(rank) else "n/a")
    p5 = row.get("pred_epm_5y")
    five = (p5 - epm_now) if (pd.notna(p5) and pd.notna(epm_now)) else None
    rcols[1].metric("5-year change", f"{five:+.2f}" if five is not None else "n/a")

    # chart: continuous actual -> projected, with a typical-error band
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
    if pd.notna(epm_now):
        fig.add_trace(go.Scatter(x=[CUR], y=[epm_now], mode="markers",
                                 marker=dict(size=13, color="#111"),
                                 name="current (actual)",
                                 hovertemplate="current %{y:+.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Season (end year)", yaxis_title="EPM",
                      height=440, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(range=Y_RANGE)
    fig.update_xaxes(tickmode="array", tickvals=yrs)
    st.plotly_chart(fig, width="stretch")
    st.caption("Solid dot = current (actual) EPM. Shaded band = typical out-of-fold "
               "absolute error at each horizon (not a confidence interval).")

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
                "EPM now": r.get("epm_now"),
                "1-Year": r.get("pred_epm_1y"),
                "3-Year": r.get("pred_epm_3y"),
                "5-Year": r.get("pred_epm_5y"),
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

    pcol = f"pred_epm_{h}y"
    b = current.dropna(subset=[pcol, "epm_now"]).copy()
    if search:
        b = b[b["player_name"].str.contains(search, case=False, na=False)]
    b = b.sort_values(pcol, ascending=False).reset_index(drop=True)
    b.insert(0, "Rank", b.index + 1)
    # change vs current EPM; blanked (NA) for players projected below -2 (deep negatives)
    b["change"] = b[pcol] - b["epm_now"]
    b.loc[b[pcol] < -2, "change"] = pd.NA
    out = b[["Rank", "player_name", "team", "age", "epm_now", pcol, "change"]].copy()
    out.columns = ["Rank", "Player", "Team", "Age", "EPM now", f"Proj {h}y", "Change"]

    st.caption(f"{len(out)} players")
    sty = out.style.format({"Age": "{:.0f}", "EPM now": "{:+.2f}", f"Proj {h}y": "{:+.2f}",
                            "Change": "{:+.2f}"}, na_rep="—")
    st.dataframe(sty, hide_index=True, width="stretch", height=640)
    st.download_button("Download CSV", out.to_csv(index=False).encode(),
                       file_name=f"epm_projections_{h}y.csv", mime="text/csv")

# ============================================================ Methodology
with tab_method:
    st.markdown("""
### What this is

A model that projects each NBA player's Estimated Plus-Minus (EPM) one to five
seasons into the future.

### How it works

- **Features:** current and lagged EPM, DARKO DPM as a second impact signal, observed
  EPM, per-36 box-score rates, draft position, team context, and minutes/impact
  interactions.
- **Leak-free validation:** every cross-validation fold trains only on seasons before
  the test seasons, so the model is never scored on data it has effectively seen.
  Reported numbers are out-of-fold.
- **Survivorship handling:** players who leave the league are not silently dropped,
  which would bias the model toward survivors. Players whose careers ended get a
  replacement-level target so decline is modeled rather than ignored.

### Limitations

- Gradient-boosted trees cannot extrapolate beyond the range seen in training, so an
  unprecedented young season is pulled toward historical precedent and the projection
  will read conservative for true outliers.
- EPM inputs begin in the early 2000s, so seasons before then are not covered.
""")
