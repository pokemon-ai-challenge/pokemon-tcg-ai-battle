#!/usr/bin/env bash
# gen2(2026-08)模倣学習データの取得ドライバ。
# 既存の replays/ + index/ (2026-07取得, 4975件) には一切触れず、
# replays_g2/ + index_g2/ に完全分離して新規リプレイを集める。
#
#   1) 上位200チーム(全エピソード、最大 TOP_MAX 件)  = elite品質(concentrated重み用)
#   2) rank 201-2000 から1チーム2件               = アーキタイプ網羅(少数アーキ用)
#
# 冪等: 途中で止めても再実行すれば download_replay() の dest.exists() と
# count_episodes_by_team() のスキップが効いて続きから進む。
set -u
cd "$(dirname "$0")"

LOG_DIR=_g2_logs
mkdir -p "$LOG_DIR" index_g2/leaderboard_history

COMMON=(--out-dir replays_g2
        --master-index-path index_g2/episodes_master.jsonl
        --leaderboard-history-dir index_g2/leaderboard_history)

TOP_MAX=${TOP_MAX:-3000}
RANK_TO=${RANK_TO:-2000}
# 1件あたり実測5.3秒(kaggle CLI起動+4.4MB転送)で逐次だと十数時間かかるため並列化する。
# ただし workers=8 + 制限なしだと team-submissions が全件 429 になったため、
# 全スレッド共有の最小間隔0.7秒(=約1.4 req/s)をかける。この設定で実測429ゼロ・約7倍速。
WORKERS=${WORKERS:-6}
DEEP_WORKERS=${DEEP_WORKERS:-6}

echo "=== [1/2] top200 fetch (max ${TOP_MAX}, workers ${WORKERS}) $(date -u +%FT%TZ) ==="
python -u fetch_top_episodes.py --top 200 --submissions-per-team 1 --workers "$WORKERS" \
  --max-episodes "$TOP_MAX" "${COMMON[@]}" 2>&1 | tee "$LOG_DIR/fetch_top200.log"
echo "=== [1/2] done rc=${PIPESTATUS[0]} files=$(ls replays_g2 2>/dev/null | wc -l) ==="

echo "=== [2/2] deep fetch rank 201-${RANK_TO} (workers ${DEEP_WORKERS}) $(date -u +%FT%TZ) ==="
python -u fetch_deep_decks.py --rank-from 201 --rank-to "$RANK_TO" --workers "$DEEP_WORKERS" \
  --episodes-per-team 2 --submissions-per-team 1 "${COMMON[@]}" 2>&1 | tee "$LOG_DIR/fetch_deep.log"
echo "=== [2/2] done rc=${PIPESTATUS[0]} files=$(ls replays_g2 2>/dev/null | wc -l) ==="

echo "=== ALL DONE $(date -u +%FT%TZ) replays_g2=$(ls replays_g2 | wc -l) master=$(wc -l < index_g2/episodes_master.jsonl) ==="
