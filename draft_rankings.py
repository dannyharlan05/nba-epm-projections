"""Draft Rankings page: predicted peak NBA EPM for classes the model never trained on.

Everything in artifacts/draft_* is written by the last cell of EPM_Draft.ipynb.
"""
import json
import os

import pandas as pd
import streamlit as st

from draft_model import GROUPS, pos_group, predict
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
    st.error("artifacts/draft_rankings.csv not found. Run the last cell of EPM_Draft.ipynb.")
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
year = c1.selectbox("Draft class", [int(y) for y in years] + ["All"])
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
            format="%+d", help="Mock rank minus model rank. + = model likes him more than the mock."),
        "Mock pick": st.column_config.NumberColumn(help="61 = on the bubble / unslotted"),
        "Class": st.column_config.NumberColumn(format="%d"),
    },
)
st.caption(f"{len(out)} players. Board is limited to players on the mock board (pick 61 or better).")


# Build a prospect: raw stats in, predicted peak EPM out (features are rebuilt in draft_model.py)
st.divider()
st.subheader("Build a prospect")

_m = load_model()
if _m is None:
    st.info("Model not exported yet. Run the last cell of EPM_Draft.ipynb.")
    st.stop()
bundle, booster = _m
stats = bundle["inputs"]
inputs = load_inputs()

LABELS = {
    "ORtg": "Offensive rating", "usg": "Usage %", "twoP_per": "2P %", "FT_per": "FT %",
    "3par": "3PA rate (3PA / FGA)", "AST_per": "Assist %", "TO_per": "Turnover %",
    "ast/tov": "Assist / turnover", "ORB_per": "Off. rebound %", "DRB_per": "Def. rebound %",
    "blk_per": "Block %", "stl_per": "Steal %",
    "ht_inches": "Height (in)", "WEIGHT": "Weight (lb)", "age": "Age (Feb 1 of draft year)",
    "projected_pick": "Mock pick (1-61)", "Rec Rank": "Recruit rating (0 = unranked)",
}
# ask only for the inputs the exported model needs (power6 is the checkbox below)
LABELS = {c: LABELS.get(c, c) for c in stats if c != "power6"}
SECTIONS = [
    ("Scoring", ["ORtg", "usg", "twoP_per", "FT_per"]),
    ("Shooting profile and playmaking", ["3par", "AST_per", "TO_per", "ast/tov"]),
    ("Rebounding and defense", ["ORB_per", "DRB_per", "blk_per", "stl_per"]),
    ("Size, age and draft slot", ["ht_inches", "WEIGHT", "age", "projected_pick"]),
    ("Recruiting", ["Rec Rank"]),
]
_placed = {c for _, cols in SECTIONS for c in cols}
_other = [c for c in LABELS if c not in _placed]
SECTIONS += [("Other inputs", _other[k:k + 4]) for k in range(0, len(_other), 4)]
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
        st.session_state[f"bp_{c}"] = (_num(row[c]) if c in row.index else None) if row is not None \
            else round(stats[c]["median"], 3)
    p6 = _num(row.get("power6")) if row is not None else stats.get("power6", {}).get("median", 1.0)
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
             format_func=lambda o: o.rsplit("  #", 1)[0])
s2.selectbox("Position group", POS, key="bp_pos",
             help="Which group's medians the stats are compared to. Auto goes by height.")
s3.button("Reset", on_click=_fill, width="stretch")


def _box(col, c):
    med = stats[c]["median"]
    small = abs(med) <= 1.5          # stored as a fraction (2P% = .52)
    col.number_input(
        LABELS[c], key=f"bp_{c}", value=None, step=0.01 if small else (1.0 if c in ("projected_pick", "WEIGHT") else 0.1),
        format="%.3f" if small else "%.1f",
        help=f"Training median {med:.3g}; typical range {stats[c]['lo']:.3g} to {stats[c]['hi']:.3g}",
    )


for title, cols in SECTIONS:
    cols = [c for c in cols if c in LABELS]
    if not cols:
        continue
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

# rank within the 'Start from' player's class, else the class picked above
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
