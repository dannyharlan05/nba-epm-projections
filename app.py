"""Streamlit front end for the EPM projection model.

Loads pre-built artifacts (run `python build_artifacts.py` first), so it never
retrains on launch. Three tabs: player projection, leaderboards, methodology.

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

# fixed y-axis range so every player's chart is on the same scale
_pred_cols = [f"pred_epm_{h}y" for h in HORIZONS]
_vals = pd.concat([df[c] for c in _pred_cols] + [df["epm_now"]]).dropna()
Y_RANGE = [float(_vals.min()) - 0.5, float(_vals.max()) + 0.5]

st.title("NBA EPM Projections")
st.caption(f"Multi-year Estimated Plus-Minus forecasts. "
           f"Current season: {CUR}. Validated with leak-free temporal cross-validation.")

tab_player, tab_board, tab_method = st.tabs(["Player", "Leaderboards", "Methodology"])

# ---------------------------------------------------------------- Player tab
with tab_player:
    current = df[df["season"] == CUR].sort_values("epm_now", ascending=False)
    names = current["player_name"].dropna().unique().tolist()
    default = names.index("Nikola Jokic") if "Nikola Jokic" in names else 0
    name = st.selectbox("Player", names, index=default)

    row = current[current["player_name"] == name].iloc[0]
    proj = [row.get(f"pred_epm_{h}y") for h in HORIZONS]
    years = [CUR + h for h in HORIZONS]

    c1, c2 = st.columns(2)
    c1.metric("Current EPM", f"{row['epm_now']:.2f}" if pd.notna(row["epm_now"]) else "n/a")
    c2.metric("Age", f"{row['age']:.0f}" if pd.notna(row["age"]) else "n/a")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=proj, name="Projected EPM",
                             mode="lines+markers",
                             line=dict(color="#1f77b4", width=3),
                             hovertemplate="%{y:.2f}"))
    if pd.notna(row["epm_now"]):
        fig.add_hline(y=row["epm_now"], line_dash="dot", line_color="gray",
                      annotation_text="current EPM")
    fig.update_layout(xaxis_title="Season (end year)", yaxis_title="Projected EPM",
                      height=420, hovermode="x unified")
    fig.update_yaxes(range=Y_RANGE)
    fig.update_xaxes(tickmode="array", tickvals=years)
    st.plotly_chart(fig, use_container_width=True)

    tbl = pd.DataFrame({"Season": years, "Projected EPM": proj})
    st.dataframe(tbl.style.format({"Projected EPM": "{:.2f}"}),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Leaderboards
with tab_board:
    col1, col2 = st.columns(2)
    h = col1.selectbox("Years ahead", HORIZONS, index=2, key="lb_h")
    search = col2.text_input("Search player")

    pcol = f"pred_epm_{h}y"
    board = current.copy()
    if search:
        board = board[board["player_name"].str.contains(search, case=False, na=False)]

    board = board[["player_name", "team", "age", "epm_now", pcol]].dropna(subset=[pcol])
    board = board.sort_values(pcol, ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", board.index + 1)
    board.columns = ["Rank", "Player", "Team", "Age", "EPM now", f"Projected {h}y"]
    st.caption(f"{len(board)} players")
    st.dataframe(
        board.style.format({"Age": "{:.0f}", "EPM now": "{:.2f}",
                            f"Projected {h}y": "{:.2f}"}),
        hide_index=True, use_container_width=True, height=700)

# ---------------------------------------------------------------- Methodology
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
