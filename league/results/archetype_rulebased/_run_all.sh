#!/usr/bin/env bash
set -u
ARCHES="marnie_grimmsnarl_ex mega_lucario_ex archaludon_ex dragapult_ex rocket_mewtwo_ex shirona_garchomp_ex crustle"
GAMES=300
SUMMARY=league/results/archetype_rulebased/_summary.txt
: > "$SUMMARY"
for A in $ARCHES; do
  DECK="kaggle_replays/meta_analysis/archetype_decks/${A}/01.csv"
  W="sample_submission/ptcg_ai/learning/policy_weights_${A}.json"
  OUT="league/results/archetype_rulebased/${A}_vs_rule_${GAMES}.json"
  echo "########## $(date +%H:%M:%S) START $A ##########"
  if python run_league.py \
       --agent-a ml_policy --weights-a "$W" --deck-a "$DECK" \
       --agent-b rule_based               --deck-b "$DECK" \
       --games "$GAMES" --progress-every 50 --out "$OUT" \
       > "league/results/archetype_rulebased/${A}_run.log" 2>&1; then
    echo "$A OK -> $OUT" >> "$SUMMARY"
    echo "########## $(date +%H:%M:%S) DONE  $A ##########"
  else
    echo "$A FAILED (exit $?)" >> "$SUMMARY"
    echo "########## $(date +%H:%M:%S) FAIL  $A ##########"
  fi
done
echo "===== RULE_BASED FLOOR BATCH COMPLETE ====="
cat "$SUMMARY"
