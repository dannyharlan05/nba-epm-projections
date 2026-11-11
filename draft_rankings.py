"""Draft Rankings page: predicted peak NBA EPM for draft classes the model never trained on.

Reads artifacts/draft_rankings.csv, written by the RANKINGS PAGE cell of Darko_Draft.ipynb
(re-run that cell after a refit, then commit the CSV).
"""
import json
import os

import pandas as pd
import streamlit as st

from draft_model import GROUPS, feature_matrix, pos_group, predict
import draft_model

ART = "artifacts"
CSV = os.path.join(ART, "draft_rankings.csv")
META = os.path.join(ART, "draft_meta.json")
INPUTS = os.path.join(ART, "draft_inputs.csv")


@st.cache_resource
def load_model():
    return draft_model.load(ART)


@st.cache_data
def load_inputs():
    return pd.read_csv(INPUTS) if os.path.exists(INPUTS) else None


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


# =========================================================
# BUILD A PROSPECT -- type in raw stats, get the model's predicted peak EPM.
# Only raw inputs are asked for; interactions (ORtg x usage, ...), stocks, Usg/Ast and the
# size-relative versions are rebuilt from them in draft_model.py, the same way the notebook does.
# =========================================================
st.divider()
st.subheader("Build a prospect")

_m = load_model()
if _m is None:
    st.info("Model not exported yet. Re-run the RANKINGS PAGE cell in Darko_Draft.ipynb; it writes "
            "artifacts/draft_model.json, draft_xgb.json and draft_inputs.csv.")
    st.stop()
bundle, booster = _m
stats = bundle["inputs"]
inputs = load_inputs()

LABELS = {
    "ORtg": "Offensive rating", "usg": "Usage %", "twoP_per": "2P %", "FT_per": "FT %",
    "AST_per": "Assist %", "TO_per": "Turnover %", "ORB_per": "Off. rebound %",
    "DRB_per": "Def. rebound %", "blk_per": "Block %", "stl_per": "Steal %",
    "ht_inches": "Height (in)", "WEIGHT": "Weight (lb)", "age": "Age (Feb 1 of draft year)",
    "projected_pick": "Mock pick (1-61)",
}
SECTIONS = [
    ("Scoring", ["ORtg", "usg", "twoP_per", "FT_per"]),
    ("Playmaking", ["AST_per", "TO_per"]),
    ("Rebounding and defense", ["ORB_per", "DRB_per", "blk_per", "stl_per"]),
    ("Size, age and draft slot", ["ht_inches", "WEIGHT", "age", "projected_pick"]),
]
POS = ["Auto (from height)", "Guard", "Wing", "Big"]
BLANK = "Blank (training medians)"


def _num(v):
    v = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(v) else float(v)


def _fill():
    """Load the 'Start from' choice into every input."""
    pick = st.session_state.get("bp_start", BLANK)
    row = None
    if inputs is not None and pick != BLANK:
        row = inputs.iloc[int(pick.rsplit("#", 1)[1])]
    for c in LABELS:
        st.session_state[f"bp_{c}"] = _num(row[c]) if row is not None else round(stats[c]["median"], 3)
    p6 = _num(row["power6"]) if row is not None else stats["power6"]["median"]
    st.session_state["bp_power6"] = bool(p6 is not None and p6 >= 0.5)
    st.session_state["bp_pos"] = str(row["grp"]).title() if row is not None and str(row["grp"]) in GROUPS else POS[0]
    st.session_state["bp_penalty"] = float(row["penalty"]) if row is not None else 0.0


if "bp_ORtg" not in st.session_state:
    _fill()

st.caption(
    "Enter raw college stats (same units as Barttorvik). The model's interaction terms, like "
    "ORtg x usage, 2P% x usage and assist% x usage, plus stocks and usage per assist, are computed "
    "from these, and every stat except age, pick and conference is compared to the median for the "
    "player's position group. Leave a box empty to treat it as missing."
)

opts = [BLANK]
if inputs is not None:
    _o = inputs.sort_values(["year", "player"], ascending=[False, True])
    opts += [f"{r.player} ({r.year}{', intl' if r.type == 'INTL' else ''})  #{i}" for i, r in _o.iterrows()]
s1, s2, s3 = st.columns([3, 2, 1])
s1.selectbox("Start from", opts, key="bp_start", on_change=_fill,
             format_func=lambda o: o.rsplit("  #", 1)[0],
             help="Prefill every box with a board player's real inputs, then tweak.")
s2.selectbox("Position group", POS, key="bp_pos",
             help="Sets which group median each stat is compared to. Auto: under 6'7\" guard, "
                  "under 6'10\" wing, else big (a guard over 6'7\" counts as a wing).")
s3.button("Reset", on_click=_fill, help="Back to the 'Start from' values", width="stretch")


def _box(col, c):
    med = stats[c]["median"]
    small = abs(med) <= 1.5          # 0-1 scaled stat (e.g. 2P% stored as .52)
    col.number_input(
        LABELS[c], key=f"bp_{c}", value=None, step=0.01 if small else (1.0 if c in ("projected_pick", "WEIGHT") else 0.1),
        format="%.3f" if small else "%.1f",
        help=f"Training median {med:.3g}; typical range {stats[c]['lo']:.3g} to {stats[c]['hi']:.3g}",
    )


for title, cols in SECTIONS:
    st.markdown(f"**{title}**")
    cc = st.columns(4)
    for i, c in enumerate(cols):
        _box(cc[i], c)
st.checkbox("High-major conference (ACC, Big Ten, Big 12, SEC, Big East)", key="bp_power6",
            help="Internationals are scored as high-major.")

raw = {c: st.session_state[f"bp_{c}"] for c in LABELS}
raw["power6"] = 1.0 if st.session_state["bp_power6"] else 0.0
_pos = st.session_state["bp_pos"]
raw["grp"] = pos_group(None if _pos == POS[0] else _pos.lower(), raw["ht_inches"])
raw_df = pd.DataFrame([raw])
pred = float(predict(raw_df, bundle, booster)[0]) - st.session_state.get("bp_penalty", 0.0)

# where he'd land in a class: the 'Start from' player's class, else the class picked above
_start = st.session_state.get("bp_start", BLANK)
cls = int(inputs.iloc[int(_start.rsplit("#", 1)[1])]["year"]) if inputs is not None and _start != BLANK else year
r1, r2, r3 = st.columns(3)
r1.metric("Predicted peak EPM", f"{pred:+.2f}")
r2.metric("Position group used", raw["grp"].title())
if cls != "All":
    _cl = df[df["year"] == cls]
    _cl = _cl[_cl["player"] != _start.rsplit(" (", 1)[0]] if _start != BLANK else _cl
    r3.metric(f"Rank in {cls} class", f"{int((_cl['pred_epm'] > pred).sum()) + 1} of {len(_cl) + 1}")
if st.session_state.get("bp_penalty", 0.0):
    st.caption(f"Includes the -{st.session_state['bp_penalty']:g} unmapped-league penalty this player gets on the board.")

with st.expander("What the model sees"):
    X = feature_matrix(raw_df, bundle)[0]
    st.dataframe(pd.DataFrame({"feature": bundle["features"], "value": X}),
                 hide_index=True, width="stretch",
                 column_config={"value": st.column_config.NumberColumn(format="%.3f")})
    st.caption("_rel = value minus the median for the position group. Blank = missing.")
