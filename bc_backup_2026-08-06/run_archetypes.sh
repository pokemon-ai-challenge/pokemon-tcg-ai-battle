#!/bin/bash
set -e
cd "C:/Users/rinnz/Documents/pokemon/pokemon-tcg-ai-battle/kaggle_replays/policy_net"
export PYTHONIOENCODING=utf-8
for arch in rocket_mewtwo_ex omatsuri_ondo shirona_garchomp_ex ogerpon_teal_ex dragapult_ex; do
  echo "==================== START $arch $(date) ===================="
  python run_archetype_pipeline.py --archetype "$arch"
  echo "==================== DONE $arch $(date) ===================="
done
echo "ALL_ARCHETYPES_DONE"
