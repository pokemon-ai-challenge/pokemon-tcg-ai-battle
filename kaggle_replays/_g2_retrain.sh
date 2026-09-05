#!/usr/bin/env bash
# インデックス復旧後の再学習。rebuild_master_index.py で rank_at_fetch を復元したので、
# concentrated 重み(rank<=20 を8倍 = 本番=構成C のレシピ)が初めて正しく効く状態になった。
# 先に走らせた11アーキの学習は rank 充足率15%(実質ほぼ均等重み)だったので作り直す。
#
#   1. 不足2アーキの狙い撃ち取得(1時間で打ち切り。クォータ次第で取れる分だけ取る)
#   2. 再ラベリング
#   3. 全11アーキを再学習(policy_weights_<arch>_g2.json を上書き)
set -u
cd "$(dirname "$0")"

LOG_DIR=_g2_logs
mkdir -p "$LOG_DIR"

REPLAYS=replays_g2
MASTER=index_g2/episodes_master.jsonl
DECK_DB=deck_predictor/output/deck_db_g2.jsonl
LABELS=deck_predictor/output/deck_labels_g2.jsonl
TOPUP_LABELS="$LOG_DIR/deck_labels_g2_topup.jsonl"
TARGETS=(archaludon_ex rocket_mewtwo_ex)

ARCHS=(alakazam marnie_grimmsnarl_ex ogerpon_teal_ex shirona_garchomp_ex omatsuri_ondo
       mega_froslass_ex mega_lucario_ex rocket_mewtwo_ex dragapult_ex archaludon_ex crustle)

# --- 1. 狙い撃ち取得(1時間で打ち切る。クォータ枯渇で無限に待たないため) ---
echo "=== [1/3] 狙い撃ち取得(上限1時間) $(date -u +%FT%TZ) ==="
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
print(f"  狙い撃ち対象 {n} 行(対象: {sorted(keep)})")
PY
timeout 3600 python -u fetch_minority_archetype_episodes.py --deck-labels "$TOPUP_LABELS" \
  --min-decks 999999 --max-episodes "${TOPUP_MAX:-400}" --max-episodes-per-team 15 \
  --submissions-per-team 3 --workers 3 --min-interval 2.5 \
  --out-dir "$REPLAYS" --master-index-path "$MASTER" 2>&1 | tee "$LOG_DIR/fetch_topup2.log"
echo "=== [1/3] done files=$(ls "$REPLAYS" | wc -l) $(date -u +%FT%TZ) ==="

# --- 2. 再ラベリング ---
echo "=== [2/3] 再ラベリング $(date -u +%FT%TZ) ==="
python -u deck_predictor/extract_decks.py --replays-dir "$REPLAYS" \
  --master-index-path "$MASTER" --out "$DECK_DB" 2>&1 | tee "$LOG_DIR/extract_decks_final.log"
python -u deck_predictor/label_decks.py --deck-db "$DECK_DB" --out "$LABELS" \
  --rough-predictor-json deck_predictor/rough_predictor_g2.json \
  --report deck_predictor/output/label_report_g2.md 2>&1 | tee "$LOG_DIR/label_decks_final.log"
python -u deck_predictor/audit_archetype_labels.py --deck-db "$DECK_DB" --deck-labels "$LABELS" \
  --rough-predictor-json deck_predictor/rough_predictor_g2.json \
  --out "$LOG_DIR/_g2_label_audit.md" 2>&1 | tee "$LOG_DIR/audit_final.log"

# --- 3. 全11アーキを再学習 ---
SUMMARY="$LOG_DIR/_g2_summary_final.txt"
: > "$SUMMARY"
for arch in "${ARCHS[@]}"; do
  echo "=== [3/3] BC pipeline: $arch $(date -u +%FT%TZ) ==="
  python policy_net/run_archetype_pipeline.py --archetype "$arch" --tag g2 \
    --replays-dir "$REPLAYS" --master-index-path "$MASTER" --deck-labels-path "$LABELS" \
    > "$LOG_DIR/pipeline_${arch}_g2_final.log" 2>&1
  if [ $? -eq 0 ]; then echo "$arch OK" | tee -a "$SUMMARY"
  else echo "$arch FAILED" | tee -a "$SUMMARY"; fi
done

echo "=== RETRAIN ALL DONE $(date -u +%FT%TZ) ==="
cat "$SUMMARY"
