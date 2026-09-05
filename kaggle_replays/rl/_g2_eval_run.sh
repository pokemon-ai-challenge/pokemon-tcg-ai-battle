#!/usr/bin/env bash
# gen2 RL候補をチャンピオン(climb)と直接比較する。
#
# 学習フィールド上の数字(g2cont 0.603 vs climb 0.550)は同じフィールドで測っているので
# 過学習の可能性がある。ここでは2つの独立な条件で確認する:
#   1. gen2フィールド(11アーキ) : 学習と同じ相手だが、評価は本番相当の config overlay で回す
#   2. july7フィールド(7アーキ)  : **学習に使っていない**相手。ここで落ちるなら新フィールドへの
#                                  過学習であり、LBへの転移は期待できない
# baseline は常にチャンピオン climb。
set -u
cd "$(dirname "$0")"

GAMES=${GAMES:-120}
WORKERS=${WORKERS:-6}

for preset in g2 july7; do
  for cand in g2cont g2bc; do
    echo "=== eval field=$preset cand=$cand vs climb $(date -u +%FT%TZ) ==="
    python -u eval_field.py --field-preset "$preset" \
      --baseline-weights policy_weights_alakazam_rl_climb.json \
      --rl-weights "policy_weights_alakazam_rl_${cand}.json" \
      --games "$GAMES" --workers "$WORKERS" 2>&1 | tail -22
  done
done
echo "=== EVAL ALL DONE $(date -u +%FT%TZ) ==="
