# NBA EPM Projections

Forecasting NBA player impact (Estimated Plus-Minus) one to five seasons into the future.

**[▶ Live demo](https://nba-epm-projections.streamlit.app/)** · interactive player projections, leaderboards, and methodology.

## Results

Out-of-fold MAE from leak-free temporal cross-validation (~12,600 player-seasons, 2001–2026):

| Horizon | OOF MAE (EPM) |
|---|---|
| 1 year | 0.86 |
| 2 years | 1.02 |
| 3 years | 1.06 |
| 4 years | 1.04 |
| 5 years | 1.04 |

For reference, single-season EPM roughly spans −6 to +8, and ~95% of rotation players
fall within about ±3, so a sub-1.1 MAE several years out is a meaningful signal.

## Why this project

A lot of amateur sports models leak future information into training, score themselves
only on the players who stuck around, and report optimistic accuracy as a result. This
project is built to avoid those traps:

| Decision | What it does | Why it matters |
|---|---|---|
| Leak-free temporal CV | Each fold trains only on seasons before the test seasons | The reported error reflects real forward forecasting, not an inflated in-sample number |
| Survivorship correction | Players who leave the league get a replacement-level target instead of being dropped | Without it, the model over-projects aging veterans, since their bad seasons disappear into retirement |

---

## What's inside

```
epm-projections/
├── src/
│   ├── data.py        # load game logs, DARKO, draft, predictive + actual EPM
│   ├── features.py    # build the per-(player, season) training table
│   └── model.py       # temporal CV, model training, OOF predictions
├── build_artifacts.py # run the pipeline once, serialize models + predictions
├── app.py             # Streamlit UI (player view, leaderboards, methodology)
├── data/              # drop raw CSVs here (git-ignored)
└── artifacts/         # generated models + predictions (git-ignored)
```

The model **trains in `build_artifacts.py` and the app only loads** the serialized
output — so the UI is fast and deployable on a free tier.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) put the raw inputs in ./data  (see "Data inputs" below)
# 2) build models + predictions
python build_artifacts.py
# 3) launch the app
streamlit run app.py
```

### Data inputs (`./data`)
| File | Source | Provides |
|---|---|---|
| `player_game_logs_clean.csv` | NBA API player game logs | box-score rates, minutes, availability, team |
| `DARKO ... Full DPM History.csv` | DARKO public history | DPM (box-prior impact) + age |
| `EPM data (N).csv` | Dunks & Threes (predictive EPM) | the primary signal + targets |
| `Dunks & Threes Stats*.csv` | Dunks & Threes (actual EPM) | observed EPM (late-season form) |
| `models_by_cluster.pkl` *(optional)* | college draft model | `draft_score` for young players |

Draft history is pulled once from the NBA API and cached to `data/draft_history.csv`.

To refresh projections after more games are played, see **[UPDATING.md](UPDATING.md)** —
in short, refresh the inputs and re-run `python build_artifacts.py`.

---

## Model details

**Target.** `target_epm_{h}y` = a player's EPM `h` seasons later (`h` = 1…5).

**Features.** Current and lagged EPM (with slope and deltas), DARKO DPM as a second
impact signal, observed EPM and its lags, per-36 box rates, draft position, team
context, and minutes-times-impact interactions. A monotonic constraint keeps
more current EPM from ever lowering a projection, all else equal.

**Validation.** `temporal_splits` builds expanding-window folds where the test seasons
always come after the training seasons. Reported predictions for past players are
out-of-fold; current-season players (no target yet) use a final model fit on
everything. Hyperparameters live in one place (`DEFAULT_PARAMS` + `PARAMS_BY_HORIZON`
in `src/model.py`) and flow to CV and prediction alike.

## Limitations

- Gradient-boosted trees cannot extrapolate beyond the training target range, so an
  unprecedented season (for example a 20-year-old at 7+ EPM) is regressed toward
  precedent and the projection reads conservative for true outliers.
- EPM inputs begin in the early 2000s, so seasons before then are not covered.

## Tech

Python, XGBoost, scikit-learn, Streamlit, Plotly, pandas.
