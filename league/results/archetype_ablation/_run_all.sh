#!/usr/bin/env bash
set -u
ARCHES="mega_lucario_ex archaludon_ex dragapult_ex marnie_grimmsnarl_ex crustle shirona_garchomp_ex rocket_mewtwo_ex"
GAMES=300
SUMMARY=results/archetype_ablation/_summary.txt
: > "$SUMMARY"
for A in $ARCHES; do
  DECK="kaggle_replays/meta_analysis/archetype_decks/${A}/01.csv"
  W="sample_submission/ptcg_ai/learning/policy_weights_${A}.json"
  OUT="results/archetype_ablation/${A}_ablation_${GAMES}.json"
  echo "########## $(date +%H:%M:%S) START $A ##########"
  if python run_league.py \
       --agent-a ml_policy --weights-a "$W"    --deck-a "$DECK" \
       --agent-b ml_policy                     --deck-b "$DECK" \
       --games "$GAMES" --progress-every 50 --out "$OUT" \
       > "results/archetype_ablation/${A}_run.log" 2>&1; then
    echo "$A OK -> $OUT" >> "$SUMMARY"
    echo "########## $(date +%H:%M:%S) DONE  $A ##########"
  else
    echo "$A FAILED (exit $?)" >> "$SUMMARY"
    echo "########## $(date +%H:%M:%S) FAIL  $A ##########"
  fi
done
echo "===== ABLATION BATCH COMPLETE ====="
cat "$SUMMARY"
