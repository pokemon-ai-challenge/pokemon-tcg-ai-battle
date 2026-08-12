#!/usr/bin/env bash
# gen2 模倣学習でチャンピオン(climb)を改善する試行。
#
# チャンピオン = abl_5_full + policy_weights_alakazam_rl_climb.json + Plan Aデッキ(LB 823-831)。
# そのRLは「7月BCを初期値」に「7月BC 7アーキのフィールド」で学習したもの。
# gen2 では 自分(初期値)と 相手(フィールド)の両方の模倣学習を2026-08版に差し替える。
#
#   A) g2bc   : 初期値 = gen2 alakazam BC(test top1 0.6626、7月BCは0.5806)
#   B) g2cont : 初期値 = チャンピオン climb 重み(RLの蓄積を保持して新フィールドへ適応)
#   共通      : 相手フィールド = gen2 11アーキ(ミラー/オーガポン/メガユキメノコ/おまつりおんど込み)
#               学習デッキ = sample_submission/deck.csv(Plan A = チャンピオンと同一)
#
# 既存の重みは1つも上書きしない(出力は policy_weights_alakazam_rl_g2bc/g2cont.json)。
set -u
cd "$(dirname "$0")"

ROOT=/c/dev/pokemon-tcg-ai-battle
DECK="$ROOT/sample_submission/deck.csv"
ITERS=${ITERS:-30}
GPI=${GPI:-384}
WORKERS=${WORKERS:-6}
EVAL_GAMES=${EVAL_GAMES:-300}
EVAL_EVERY=${EVAL_EVERY:-5}

run () {  # $1=tag $2=初期重み
  echo "=== RL $1 (初期値 $2) $(date -u +%FT%TZ) ==="
  python -u train_field.py --field-preset g2 \
    --learner-weights "$2" --learner-deck-path "$DECK" \
    --iters "$ITERS" --games-per-iter "$GPI" --eval-games "$EVAL_GAMES" \
    --eval-every "$EVAL_EVERY" --workers "$WORKERS" --device cpu --tag "$1"
  echo "=== RL $1 done rc=$? $(date -u +%FT%TZ) ==="
}

run g2bc   policy_weights_alakazam_g2.json
run g2cont policy_weights_alakazam_rl_climb.json

echo "=== ALL RL DONE $(date -u +%FT%TZ) ==="
