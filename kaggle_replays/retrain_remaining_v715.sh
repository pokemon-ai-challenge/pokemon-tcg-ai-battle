#!/bin/bash
# 2026-08-14: marnie_grimmsnarl_ex / rocket_mewtwo_ex / omatsuri_ondo /
# shirona_garchomp_ex / alakazam を現行715次元エンコーダへ再学習する。
# §8-2 の標準3ステップ(extract -> build_features -> train)をアーキタイプごとに実行。
# 実行後、成否をアーカイブ全体で見るには archetype_runs/<ARCH>_v715_s42_metrics.json を見る。
set -uo pipefail

REPO="C:/Users/rinnz/Documents/pokemon/pokemon-tcg-ai-battle"
ARCHES="marnie_grimmsnarl_ex rocket_mewtwo_ex omatsuri_ondo shirona_garchomp_ex alakazam"

for ARCH in $ARCHES; do
  echo "=== [$(date '+%H:%M:%S')] START $ARCH ==="

  cd "$REPO/kaggle_replays" || exit 1
  python extract_policy_dataset.py --archetype "$ARCH" \
    --out "training_data/policy_positions_${ARCH}.jsonl.gz" \
    --audit --audit-out "policy_net/audit_${ARCH}.md"
  ec=$?
  if [ $ec -ne 0 ]; then
    echo "=== [$(date '+%H:%M:%S')] EXTRACT FAILED $ARCH (exit $ec) ==="
    continue
  fi
  echo "=== [$(date '+%H:%M:%S')] EXTRACT DONE $ARCH ==="

  cd "$REPO/kaggle_replays/policy_net" || exit 1
  python build_features.py \
    --in "../training_data/policy_positions_${ARCH}.jsonl.gz" \
    --out "features_${ARCH}_v715.npz" \
    --weight-scheme concentrated \
    --deck-csv "../meta_analysis/archetype_decks/${ARCH}" \
    --opponent-vocab-dir "../meta_analysis/archetype_decks"
  ec=$?
  if [ $ec -ne 0 ]; then
    echo "=== [$(date '+%H:%M:%S')] BUILD_FEATURES FAILED $ARCH (exit $ec) ==="
    continue
  fi
  echo "=== [$(date '+%H:%M:%S')] BUILD_FEATURES DONE $ARCH ==="

  python train.py --features "features_${ARCH}_v715.npz" \
    --out-weights "$REPO/sample_submission/ptcg_ai/learning/policy_weights_${ARCH}_v715_s42.json" \
    --ablate-features self_board_card_slot,self_discard_card_slot,opp_board_card_slot,opp_discard_card_slot \
    --seed 42 --hidden-size 32 \
    --metrics-out "archetype_runs/${ARCH}_v715_s42_metrics.json"
  ec=$?
  if [ $ec -ne 0 ]; then
    echo "=== [$(date '+%H:%M:%S')] TRAIN FAILED $ARCH (exit $ec) ==="
    continue
  fi
  echo "=== [$(date '+%H:%M:%S')] TRAIN DONE $ARCH -> policy_weights_${ARCH}_v715_s42.json ==="
done

echo "=== ALL DONE ==="
