# Updating the projections (when more games are played)

The app only ever displays what's in `artifacts/`. To refresh the projections you
(1) refresh the raw inputs in `data/`, then (2) re-run `build_artifacts.py`, then
(3) restart the app. That's it. The detail below is just *where each input comes
from* and the two cases (mid-season vs. new season).

---

## The inputs and where they come from

| File in `data/` | How it updates | Who refreshes it |
|---|---|---|
| `player_game_logs_clean.csv` | re-pulled from the NBA API + cleaned | **the notebook** (`longterm_darko.ipynb`) |
| `EPM data (N).csv` | re-downloaded from Dunks & Threes | **you** (manual download) |
| `Dunks & Threes Stats*.csv` | re-downloaded from Dunks & Threes | **you** (manual download) |
| `DARKO ... Full DPM History.csv` | re-downloaded from DARKO | **you** (manual download) |
| `draft_history.csv` | auto-pulled from NBA API | the repo (cached) |

Because `data/` holds **symlinks** to the files in your main project folder, anything
you re-pull or re-download in the main folder flows through automatically — you don't
copy anything into `epm-projections/`.

---

## Case A — mid-season refresh (most common)

Some games happened this week and you want updated numbers. Nothing new is *created*,
the same files just get fresher data.

1. **Refresh game logs** — in `longterm_darko.ipynb`, run the game-log pull (the
   `refresh_current_season` cell, then the pull cell) so `player_game_logs_clean.csv`
   gets the new games. *(This is the only thing the notebook is still needed for.)*

2. **Re-download the current-season data files** from their sources and overwrite the
   ones in your main folder:
   - the **current-season EPM file** (the highest-numbered `EPM data (N).csv`)
   - the **current-season Dunks & Threes Stats file**
   - the **DARKO** history CSV

3. **Rebuild:**
   ```bash
   cd epm-projections
   python build_artifacts.py
   ```

4. **Restart the app** (or press **C** → *Clear cache* in the running app):
   ```bash
   streamlit run app.py
   ```

✅ Done. New projections everywhere.

---

## Case B — brand-new season

A new season starts, so a **new** file appears (e.g. `EPM data (27).csv`,
`Dunks & Threes Stats (25).csv`). The symlinks were made for the files that existed
before, so you must link the new ones once.

1. Do everything in **Case A** (pull logs, download new season's files).

2. **Re-link** so the loaders see the new filenames:
   ```bash
   cd epm-projections/data
   for f in "../../EPM data"*.csv "../../Dunks & Threes Stats"*.csv; do ln -sf "$f" .; done
   cd ..
   ```

3. **If there's a new rookie class**, drop the draft cache so it re-pulls:
   ```bash
   rm data/draft_history.csv
   ```

4. **Rebuild + restart:**
   ```bash
   python build_artifacts.py
   streamlit run app.py
   ```

---

## Quick reference

After refreshing the raw files in `./data` (game logs via the notebook; EPM / Dunks /
DARKO downloads), rebuild:

```bash
cd epm-projections
python build_artifacts.py
```

| Situation | Command |
|---|---|
| Games played / new season's files | `python build_artifacts.py` |
| New rookies | `rm data/draft_history.csv` then `python build_artifacts.py` |
| Changed model params (`src/model.py`) | `python build_artifacts.py` |

After any rebuild, restart the app (or Clear cache) to see the new numbers.

---

## Sanity check after a rebuild

`build_artifacts.py` prints the CV MAE table — it should stay roughly:

```
 horizon  train_mae  cv_mae
       1      ~0.69    ~0.87
       2      ~0.77    ~1.02
       3      ~0.77    ~1.06
       4      ~0.76    ~1.05
       5      ~0.72    ~1.04
```

If `cv_mae` jumps a lot, something in the data didn't load right (a download was
truncated, a season is missing). The `df_train` shape it prints (~12.6k rows) is
another quick tell — a big drop means an input file is missing or malformed.
