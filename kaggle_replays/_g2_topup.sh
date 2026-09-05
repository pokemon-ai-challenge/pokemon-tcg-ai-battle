#!/usr/bin/env bash
# gen2 で不足しているアーキタイプだけを狙い撃ちで追加取得し、その2つだけ再学習する。
#
# 背景: 2026-08-12 01:05頃に Kaggle の GetEpisodeReplay がクォータ枯渇(429)。
# 上位200チームの広域取得(残り約1000件)と深層取得(rank201-2000で約3600件)は
# クォータ効率が悪い——1981試合=3962デッキの時点で11アーキ中9つは既に十分な量があり、
# 不足は archaludon_ex(57) と rocket_mewtwo_ex(26) の2つだけ。比例配分の広域取得では
# この2つは+30/+13しか増えないので、少数クラス狙い撃ち(そのアーキを使っていたチームの
# 他の試合を引く)に切り替える。
#
# 手順:
#   1. クォータ回復までプローブ(5分間隔)
#   2. 対象2アーキだけに絞った deck_labels を作り、狙い撃ち取得
#   3. 本体の11アーキ学習(_g2_train.sh)の完了を待つ
#   4. 再ラベリングして、対象2アーキだけ再学習
set -u
cd "$(dirname "$0")"

LOG_DIR=_g2_logs
mkdir -p "$LOG_DIR"
SCRATCH="${SCRATCH:-$LOG_DIR}"

REPLAYS=replays_g2
MASTER=index_g2/episodes_master.jsonl
DECK_DB=deck_predictor/output/deck_db_g2.jsonl
LABELS=deck_predictor/output/deck_labels_g2.jsonl
TOPUP_LABELS="$SCRATCH/deck_labels_g2_topup.jsonl"
TARGETS=(archaludon_ex rocket_mewtwo_ex)
PROBE_EPISODE=${PROBE_EPISODE:-91986161}

# --- 1. クォータ回復待ち ---
echo "=== [1/4] クォータ回復待ち(5分間隔プローブ) $(date -u +%FT%TZ) ==="
i=0
until kaggle competitions replay "$PROBE_EPISODE" -p "$REPLAYS" -q >/dev/null 2>&1; do
  i=$((i + 1))
  echo "  $(date -u +%H:%M:%SZ) probe $i: まだ429"
  sleep 300
done
echo "=== クォータ回復 $(date -u +%FT%TZ) (失敗プローブ ${i}回) ==="

# --- 2. 対象アーキだけに絞って狙い撃ち取得 ---
echo "=== [2/4] 狙い撃ち取得: ${TARGETS[*]} $(date -u +%FT%TZ) ==="
python - "$LABELS" "$TOPUP_LABELS" "${TARGETS[@]}" <<'PY'
import json, sys
src, dst, *targets = sys.argv[1:]
keep = set(targets)
n = 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row["archetype"] in keep:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
print(f"  狙い撃ち対象 {n} 行を {dst} に書き出し(対象: {sorted(keep)})")
PY

# min-decks を大きくして「この2クラスは常に少数クラス」とみなさせる。
# workers/min-interval はクォータ再枯渇を避けるため本体より保守的にする。
python -u fetch_minority_archetype_episodes.py --deck-labels "$TOPUP_LABELS" \
  --min-decks 999999 --max-episodes "${TOPUP_MAX:-600}" --max-episodes-per-team 20 \
  --submissions-per-team 3 --workers 3 --min-interval 2.5 \
  --out-dir "$REPLAYS" --master-index-path "$MASTER" 2>&1 | tee "$LOG_DIR/fetch_topup.log"
echo "=== [2/4] done files=$(ls "$REPLAYS" | wc -l) $(date -u +%FT%TZ) ==="

# --- 3. 本体学習の完了待ち(同じCPUとラベルファイルを奪い合わないため) ---
echo "=== [3/4] 本体11アーキ学習の完了待ち $(date -u +%FT%TZ) ==="
until [ "$(grep -cE '(OK|FAILED)$' "$LOG_DIR/_g2_summary.txt" 2>/dev/null || echo 0)" -ge 11 ]; do
  sleep 60
done
echo "=== 本体学習 完了 $(date -u +%FT%TZ) ==="

# --- 4. 再ラベリング + 対象2アーキだけ再学習 ---
echo "=== [4/4] 再ラベリング + 再学習 $(date -u +%FT%TZ) ==="
python -u deck_predictor/extract_decks.py --replays-dir "$REPLAYS" \
  --master-index-path "$MASTER" --out "$DECK_DB" 2>&1 | tee "$LOG_DIR/extract_decks_topup.log"
python -u deck_predictor/label_decks.py --deck-db "$DECK_DB" --out "$LABELS" \
  --rough-predictor-json deck_predictor/rough_predictor_g2.json \
  --report deck_predictor/output/label_report_g2.md 2>&1 | tee "$LOG_DIR/label_decks_topup.log"

for arch in "${TARGETS[@]}"; do
  echo "=== 再学習: $arch $(date -u +%FT%TZ) ==="
  python policy_net/run_archetype_pipeline.py --archetype "$arch" --tag g2 \
    --replays-dir "$REPLAYS" --master-index-path "$MASTER" --deck-labels-path "$LABELS" \
    > "$LOG_DIR/pipeline_${arch}_g2_topup.log" 2>&1
  if [ $? -eq 0 ]; then echo "$arch OK (topup)" | tee -a "$LOG_DIR/_g2_summary.txt"
  else echo "$arch FAILED (topup)" | tee -a "$LOG_DIR/_g2_summary.txt"; fi
done

echo "=== TOPUP ALL DONE $(date -u +%FT%TZ) ==="
