# NBA EPM Projections

Projects an NBA player's EPM (Estimated Plus-Minus, a catch-all impact metric) out 1 to
5 seasons.

[Live demo](https://nba-epm-projections.streamlit.app/): pick a player, browse the
leaderboards, read how it works.

## Results

Two models, both validated with time-ordered cross-validation over ~12,600
player-seasons (2001-2026). The app has a toggle to switch between them.

| Horizon | Predictive EPM (stabilized) | Observed EPM (raw season) |
|---|---|---|
| 1 year | 0.74 | 1.12 |
| 2 years | 0.84 | 1.11 |
| 3 years | 0.84 | 1.07 |
| 4 years | 0.83 | 1.03 |
| 5 years | 0.80 | 0.98 |

EPM mostly runs from about -6 to +8. Predictive EPM is the smoothed next-day number, so
it projects well (sub-point error). Observed EPM is the raw full-season result, which
swings with injuries and form, so it carries a higher, irreducible error -- which is the
whole reason a stabilized metric exists.

## The parts I cared about getting right

Sports models cheat without meaning to, so most of the work went into not doing that:

- The CV is strictly time-ordered. Train on the past, test on the future, never the
  other way. So the error numbers above aren't inflated by leakage.
- Players who wash out of the league aren't dropped, their future is decayed from their
  last level toward replacement over the horizon (a gradual fade, not a cliff). If you
  drop them the model decides 35-year-olds age gracefully, because all their bad seasons
  turned into retirements instead.
- Predictions don't depend on row or column order, so reruns give the same numbers.

## Layout

```
epm-projections/
├── src/
│   ├── data.py        # load game logs, DARKO, draft, predictive + actual EPM
│   ├── features.py    # build the per-(player, season) training table
│   └── model.py       # temporal CV, model training, OOF predictions
├── build_artifacts.py # run the pipeline once, save models + predictions
├── app.py             # Streamlit UI (player view, leaderboards, methodology)
├── data/              # raw CSVs go here (git-ignored)
└── artifacts/         # saved models + predictions
```

Everything trains in `build_artifacts.py`. The app just reads the saved files, so it
loads instantly and runs on Streamlit's free tier.

## Running it

```bash
pip install -r requirements.txt

# 1) drop the raw inputs in ./data (see below)
# 2) build models + predictions
python build_artifacts.py
# 3) launch the app
streamlit run app.py
```

### Data

| File | Source | Provides |
|---|---|---|
| `player_game_logs_clean.csv` | NBA API player game logs | box-score rates, minutes, availability, team |
| `DARKO ... Full DPM History.csv` | DARKO public history | DPM (box-prior impact) and age |
| `EPM data (N).csv` | Dunks & Threes (predictive EPM) | the main signal and the targets |
| `Dunks & Threes Stats*.csv` | Dunks & Threes (actual EPM) | observed EPM (late-season form) |
| `models_by_cluster.pkl` *(optional)* | my college draft model | `draft_score` for young players |

Draft history pulls from the NBA API once and caches to `data/draft_history.csv`. When
new games happen, refresh the inputs and rerun `build_artifacts.py` ([UPDATING.md](UPDATING.md)
has the details).

## The model

One XGBoost model per horizon (1 through 5 years out), each predicting EPM that many
seasons later. There are two parallel versions with the same features and setup: one
targets **predictive EPM** (the stabilized next-day number, easier to project) and one
targets **observed EPM** (the raw full-season result, noisier). The app toggles between
them, and the higher error on observed EPM is exactly what you'd expect from a raw
metric versus a smoothed one.

Features are current and past EPM (plus its slope and deltas), DARKO DPM as a second
opinion on impact, the actual observed EPM and its lags, per-36 box stats, draft slot,
a bit of team context, and a couple of minutes-by-impact interactions. There's a
monotonic constraint so higher current EPM can't pull a projection down.

Validation is the expanding-window setup from above. Current players don't have a real
answer yet, so they're scored by a model fit on all the history. The hyperparameters
sit in one place in `model.py` so the CV and the live numbers can't drift apart.

## What it doesn't do well

- Trees can't predict outside the range they trained on, so something unprecedented (a
  20-year-old at +7 EPM) gets dragged back toward normal and the projection underrates
  real outliers.
- EPM data only goes back to the early 2000s.

## Stack

Python, XGBoost, scikit-learn, Streamlit, Plotly, pandas.
