#!/usr/bin/env bash
set -u
OUTDIR=../results/crustle_matchups
mkdir -p "$OUTDIR"
GAMES=300
SUMMARY="$OUTDIR/_summary.txt"
: > "$SUMMARY"
CRUSTLE_DECK="kaggle_replays/meta_analysis/archetype_decks/crustle/01.csv"
CRUSTLE_W="sample_submission/ptcg_ai/learning/policy_weights_crustle.json"

# opponent: "name deckarch weightsarg"  (weightsarg 空 = alakazam本番重み=default)
run_one () {
  local NAME="$1" DECKARCH="$2" WARG="$3"
  local DECK="kaggle_replays/meta_analysis/archetype_decks/${DECKARCH}/01.csv"
  local OUT="results/crustle_matchups/crustle_vs_${NAME}_${GAMES}.json"
  echo "########## $(date +%H:%M:%S) START crustle vs $NAME ##########"
  if python run_league.py \
       --agent-a ml_policy --weights-a "$CRUSTLE_W" --deck-a "$CRUSTLE_DECK" \
       --agent-b ml_policy $WARG --deck-b "$DECK" \
       --games "$GAMES" --progress-every 100 --out "$OUT" \
       > "$OUTDIR/crustle_vs_${NAME}_run.log" 2>&1; then
    echo "crustle_vs_${NAME} OK" >> "$SUMMARY"
  else
    echo "crustle_vs_${NAME} FAILED (exit $?)" >> "$SUMMARY"
  fi
}

run_one mega_lucario_ex      mega_lucario_ex      "--weights-b sample_submission/ptcg_ai/learning/policy_weights_mega_lucario_ex.json"
run_one archaludon_ex        archaludon_ex        "--weights-b sample_submission/ptcg_ai/learning/policy_weights_archaludon_ex.json"
run_one dragapult_ex         dragapult_ex         "--weights-b sample_submission/ptcg_ai/learning/policy_weights_dragapult_ex.json"
run_one marnie_grimmsnarl_ex marnie_grimmsnarl_ex "--weights-b sample_submission/ptcg_ai/learning/policy_weights_marnie_grimmsnarl_ex.json"
run_one rocket_mewtwo_ex     rocket_mewtwo_ex     "--weights-b sample_submission/ptcg_ai/learning/policy_weights_rocket_mewtwo_ex.json"
run_one shirona_garchomp_ex  shirona_garchomp_ex  "--weights-b sample_submission/ptcg_ai/learning/policy_weights_shirona_garchomp_ex.json"
run_one alakazam             alakazam             ""
echo "===== CRUSTLE MATCHUPS COMPLETE ====="
cat "$SUMMARY"
