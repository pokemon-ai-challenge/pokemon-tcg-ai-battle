#!/usr/bin/env bash
set -u
ARCHES="archaludon_ex crustle dragapult_ex marnie_grimmsnarl_ex rocket_mewtwo_ex shirona_garchomp_ex"
SUMMARY=archetype_runs/_batch_summary.txt
: > "$SUMMARY"
for A in $ARCHES; do
  echo "########## $(date +%H:%M:%S) START $A ##########"
  if python run_archetype_pipeline.py --archetype "$A" > "archetype_runs/_driver_${A}.log" 2>&1; then
    echo "$A OK" >> "$SUMMARY"
    echo "########## $(date +%H:%M:%S) DONE  $A ##########"
  else
    echo "$A FAILED (exit $?)" >> "$SUMMARY"
    echo "########## $(date +%H:%M:%S) FAIL  $A ##########"
  fi
done
echo "===== BATCH COMPLETE ====="
cat "$SUMMARY"
