"""Draft Rankings page: predicted peak NBA EPM for draft classes the model never trained on.

Reads artifacts/draft_rankings.csv, written by the RANKINGS PAGE cell of Darko_Draft.ipynb
(re-run that cell after a refit, then commit the CSV).
"""
import json
import os

import pandas as pd
import streamlit as st

ART = "artifacts"
CSV = os.path.join(ART, "draft_rankings.csv")
META = os.path.join(ART, "draft_meta.json")


@st.cache_data
def load():
    df = pd.read_csv(CSV)
    meta = {}
    if os.path.exists(META):
        with open(META) as f:
            meta = json.load(f)
    return df, meta


st.title("Draft Rankings")

if not os.path.exists(CSV):
    st.error("artifacts/draft_rankings.csv not found. Run the rankings cell in Darko_Draft.ipynb.")
    st.stop()

df, meta = load()
train = meta.get("train_years")
st.caption(
    "Predicted peak NBA EPM over a player's first six seasons, for draft classes the model never "
    "trained on" + (f" (trained on {train[0]}-{train[-1]})." if train else ".")
)

st.info(
    "**No athletic testing, by design.** Combine drills (verticals, agility, sprints) are left out "
    "of the model on purpose. Plenty of top prospects sit out testing, so the data is missing for "
    "exactly the players who matter most, and filling those gaps in would mostly add noise. Size "
    "still counts: height and weight are inputs. The trade-off is that athleticism only shows up "
    "indirectly, through production like blocks, steals and rebounding. A player whose college "
    "numbers outrun his physical tools may be rated too high, and an elite athlete whose production "
    "hasn't caught up yet may be rated too low. Weigh the athletic profile on top of the number."
)

years = sorted(df["year"].unique(), reverse=True)
c1, c2, c3 = st.columns([2, 2, 3])
year = c1.selectbox("Draft class", [int(y) for y in years] + ["All"])   # opens on the newest class
kind = c2.segmented_control("Type", ["All", "College", "International"], default="All")
query = c3.text_input("Search player or school", placeholder="e.g. Boozer, Duke")

f = df if year == "All" else df[df["year"] == year]
if kind == "College":
    f = f[f["type"] == "college"]
elif kind == "International":
    f = f[f["type"] == "INTL"]
if query:
    q = query.lower()
    f = f[f["player"].str.lower().str.contains(q, regex=False)
          | f["school"].fillna("").str.lower().str.contains(q, regex=False)]

f = f.sort_values("pred_epm", ascending=False).reset_index(drop=True)
out = pd.DataFrame({
    "#": range(1, len(f) + 1),
    "Player": f["player"] + f["type"].map({"INTL": "  (intl)"}).fillna(""),
    "School / league": f["school"],
    "Class": f["year"],
    "Mock pick": f["mock_pick"],
    "Model rank": f["model_rank"],
    "vs mock": f["mock_rank"] - f["model_rank"],
    "Pred peak EPM": f["pred_epm"],
})
if year != "All":
    out = out.drop(columns="Class")

st.dataframe(
    out,
    hide_index=True,
    width="stretch",
    height=min(38 * (len(out) + 1), 900),
    column_config={
        "Pred peak EPM": st.column_config.NumberColumn(format="%+.2f"),
        "vs mock": st.column_config.NumberColumn(
            format="%+d", help="Mock rank minus model rank within the class. + means the model likes him more than the mock board."),
        "Mock pick": st.column_config.NumberColumn(help="61 = on the bubble / unslotted"),
        "Class": st.column_config.NumberColumn(format="%d"),
    },
)
st.caption(f"{len(out)} players. Board is limited to players on the mock board (pick 61 or better).")
