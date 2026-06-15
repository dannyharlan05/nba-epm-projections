#!/usr/bin/env bash
# Refresh projections after more games are played.
#   ./update.sh            -> relink data + rebuild
#   ./update.sh --rookies  -> also drop the draft cache so new rookies are pulled
#
# Assumes you've already refreshed the raw files in the main project folder
# (game logs via the notebook; EPM / Dunks / DARKO downloads).
set -euo pipefail
cd "$(dirname "$0")"

echo "==> linking data files (picks up any new season's files)"
mkdir -p data
( cd data
  ln -sf "../../player_game_logs_clean.csv" .
  ln -sf "../../DARKO - Daily Adjusted and Regressed Kalman Optimized projections - Full DPM History.csv" .
  ln -sf "../../models_by_cluster.pkl" . 2>/dev/null || true
  for f in "../../EPM data"*.csv "../../Dunks & Threes Stats"*.csv; do ln -sf "$f" .; done
)

if [[ "${1:-}" == "--rookies" ]]; then
  echo "==> dropping draft cache (will re-pull new rookie class)"
  rm -f data/draft_history.csv
fi

echo "==> rebuilding projections"
python build_artifacts.py

echo "==> done. restart the app to see new numbers:  streamlit run app.py"
