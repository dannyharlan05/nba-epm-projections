"""Streamlit front end for the EPM projection models.

Loads pre-built artifacts (the notebook's export / build_artifacts.py), so it never
retrains on launch. A sidebar toggle switches the projection target between the
predictive (stabilized) EPM model and the observed (raw-season) EPM model.

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

has_actual = "pred_epm_actual_1y" in df.columns and "cv_mae_actual" in meta

st.title("NBA EPM Projections")
st.caption(f"Multi-year Estimated Plus-Minus forecasts · current season {CUR} · "
           "time-ordered cross-validation")

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
current = df[df["season"] == CUR].copy()
current["league_rank"] = current[NOW].rank(ascending=False, method="min")
NAMES = current.dropna(subset=[NOW]).sort_values(NOW, ascending=False)["player_name"].dropna().unique().tolist()


def proj_values(row):
    """[current, +1y ... +5y] and the matching seasons."""
    yrs = [CUR] + [CUR + h for h in HORIZONS]
    vals = [row.get(NOW)] + [row.get(pcolname(h)) for h in HORIZONS]
    return yrs, vals


tab_board, tab_player, tab_compare, tab_method = st.tabs(
    ["Leaderboards", "Player", "Compare", "Methodology"])

# ============================================================ Player
with tab_player:
    default = NAMES.index("Nikola Jokic") if "Nikola Jokic" in NAMES else 0
    name = st.selectbox("Player", NAMES, index=default)
    row = current[current["player_name"] == name].iloc[0]

    age = f"Age {row['age']:.0f}" if pd.notna(row.get("age")) else ""
    st.markdown(f"### {name} · {age}")

    now_val = row[NOW]

    cols = st.columns(4)
    cols[0].metric("Current EPM", f"{now_val:+.2f}" if pd.notna(now_val) else "n/a")
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
    b["change"] = (b[pcol] - b[NOW]).astype(float)
    b.loc[b[pcol] < -2, "change"] = float("nan")
    b = b.sort_values(pcol, ascending=False, na_position="last").reset_index(drop=True)
    b.insert(0, "Rank", b.index + 1)
    out = b[["Rank", "player_name", "team", "age", NOW, pcol, "change"]].copy()
    out.columns = ["Rank", "Player", "Team", "Age", "EPM now", f"Proj {h}y", "Change"]

    st.caption(f"{len(out)} players · blank Change = projected below -2")
    st.dataframe(
        out, hide_index=True, width="stretch", height=640,
        column_config={
            "Age": st.column_config.NumberColumn(format="%d"),
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

# ============================================================ Methodology
with tab_method:
    pmae = ", ".join(f"{r['cv_mae']:.2f}" for r in meta["cv_mae"])
    amae = ", ".join(f"{r['cv_mae']:.2f}" for r in meta["cv_mae_actual"]) if has_actual else "n/a"
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
  EPM, per-36 box-score rates, draft position, team context, and minutes/impact
  interactions. Both models share the same feature set.
- **Time-ordered validation:** every cross-validation fold trains only on seasons before
  the test seasons, so no future information reaches the feature set. Reported numbers are
  out-of-fold.
- **Survivorship handling:** players who leave the league are not silently dropped,
  which would bias the model toward survivors. Their future is decayed from their last
  level toward replacement over the horizon (a gradual fade, not an instant cliff), so
  decline is modeled rather than ignored. The same handling is applied to both targets.

### Limitations

- Gradient-boosted trees cannot extrapolate beyond the range seen in training, so an
  unprecedented young season is pulled toward historical precedent and the projection
  will read conservative for true outliers.
- EPM inputs begin in the early 2000s, so seasons before then are not covered.
""")
