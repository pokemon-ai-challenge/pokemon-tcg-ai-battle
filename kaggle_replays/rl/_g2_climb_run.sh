#!/usr/bin/env bash
# climb と同じレシピ(train_league.py = リーグ自己対戦RL)を gen2 データで作り直す。
#
# climb の実物メタ(policy_weights_alakazam_rl_climb.json)から復元したレシピ:
#   rl_league=True / rl_field_frac=0.45 / rl_best_iter=57 / rl_champions_final=2
#   = 固定フィールド45% + 自分の過去スナップショット(champion)55% の対戦で PPO。
#
# gen2 版で差し替えるもの:
#   - 学習の初期値      : 7月BC -> gen2 BC(policy_weights_alakazam_g2.json)
#   - 固定フィールド    : --field-preset g2(2026-08実測の11アーキ、相手は gen2 模倣ポリシー)
#   - 相手デッキ        : archetype_decks_g2/(オーガポンは旧型でなく現行型)
# 据え置くもの:
#   - 学習デッキ        : sample_submission/deck.csv(Plan A = チャンピオンと同一)
#   - ミラー相手        : climb 重み。「実ラダーのミラーは強い」という既存の知見に従う
#                         (--mirror-weights で gen2 BC に差し替え可能)
#   - ハイパーパラメータ: field_frac 0.45 ほかレシピ既定
#
# 既存の重みは1つも上書きしない(出力 policy_weights_alakazam_rl_g2climb.json)。
# 中断しても --resume で _ckpt_alakazam_g2climb.pt から再開できる。
set -u
cd "$(dirname "$0")"

ROOT=/c/dev/pokemon-tcg-ai-battle
ITERS=${ITERS:-80}
GPI=${GPI:-512}
WORKERS=${WORKERS:-7}

echo "=== league RL g2climb start $(date -u +%FT%TZ) (iters=$ITERS games/iter=$GPI workers=$WORKERS) ==="
python -u train_league.py \
  --field-preset g2 \
  --learner-weights policy_weights_alakazam_g2.json \
  --learner-deck-path "$ROOT/sample_submission/deck.csv" \
  --field-frac 0.45 \
  --iters "$ITERS" --games-per-iter "$GPI" \
  --eval-games 400 --eval-every 3 --ckpt-every 10 \
  --workers "$WORKERS" --device cpu --tag g2climb
echo "=== league RL g2climb done rc=$? $(date -u +%FT%TZ) ==="
