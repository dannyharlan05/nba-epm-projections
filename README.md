# NBA EPM Projections

Projects each NBA player's EPM 1 to 5 seasons out. Site: https://nba-epm-projections.streamlit.app/

One XGBoost model per horizon. Inputs are current and past EPM, DARKO DPM, per-36 box stats and some
team context. There are two versions: one projects predictive EPM (the smoothed daily number), the
other observed EPM (the raw season result, noisier).

CV is time-ordered, so every fold trains only on earlier seasons. Players who leave the league aren't
dropped; their projection fades toward replacement level.

Out-of-fold MAE:

| Years out | Predictive EPM | Observed EPM |
|---|---|---|
| 1 | 0.74 | 1.12 |
| 2 | 0.84 | 1.12 |
| 3 | 0.84 | 1.08 |
| 4 | 0.83 | 1.04 |
| 5 | 0.80 | 0.99 |

The Draft Rankings page is a separate model built in `EPM_Draft.ipynb` (parent folder). Its last cell
writes the `artifacts/draft_*` files.

## Running it

```bash
pip install -r requirements.txt
python build_artifacts.py
streamlit run app.py
```

`build_artifacts.py` needs these in `data/`: `player_game_logs_clean.csv` (NBA API game logs),
the DARKO DPM history CSV, and the Dunks & Threes `EPM data*.csv` and `Dunks & Threes Stats*.csv`
files. See UPDATING.md for refreshing them.
