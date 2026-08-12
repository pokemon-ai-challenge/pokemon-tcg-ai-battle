#!/usr/bin/env bash
# gen2(2026-08取得)データで11アーキタイプの模倣ポリシーを学習する。
# 前提: _g2_fetch.sh が replays_g2/ + index_g2/episodes_master.jsonl を作り終えていること。
#
# 旧成果物(policy_weights.json / policy_weights_<arch>.json / deck_labels.jsonl /
# features_<arch>.npz / policy_positions_<arch>.jsonl.gz)には一切書き込まない。
# 生成物は全て _g2 接尾辞:
#   deck_predictor/output/deck_db_g2.jsonl, deck_labels_g2.jsonl, label_report_g2.md
#   training_data/policy_positions_<arch>_g2.jsonl.gz
#   policy_net/features_<arch>_g2.npz
#   sample_submission/ptcg_ai/learning/policy_weights_<arch>_g2.json
set -u
cd "$(dirname "$0")"

LOG_DIR=_g2_logs
mkdir -p "$LOG_DIR"

REPLAYS=replays_g2
MASTER=index_g2/episodes_master.jsonl
DECK_DB=deck_predictor/output/deck_db_g2.jsonl
LABELS=deck_predictor/output/deck_labels_g2.jsonl

ARCHS=(alakazam marnie_grimmsnarl_ex ogerpon_teal_ex shirona_garchomp_ex omatsuri_ondo
       mega_froslass_ex mega_lucario_ex rocket_mewtwo_ex dragapult_ex archaludon_ex crustle)

label_pass () {  # $1 = ラベリングのパス名(ログ用)
  echo "=== label pass '$1' $(date -u +%FT%TZ) ==="
  python deck_predictor/extract_decks.py --replays-dir "$REPLAYS" \
    --master-index-path "$MASTER" --out "$DECK_DB" 2>&1 | tee "$LOG_DIR/extract_decks_$1.log"
  # gen2専用の定義でラベリングする(オーガポンの現行型が other / kamitsuorochi_ex に
  # 落ちる問題の修正版。本番 rough_predictor.json は不変)。
  python deck_predictor/build_rough_predictor_g2.py 2>&1 | tee "$LOG_DIR/build_predictor_$1.log"
  python deck_predictor/label_decks.py --deck-db "$DECK_DB" --out "$LABELS" \
    --rough-predictor-json deck_predictor/rough_predictor_g2.json \
    --report deck_predictor/output/label_report_g2.md 2>&1 | tee "$LOG_DIR/label_decks_$1.log"
}

# --- Phase A: 取得済み gen2 プールをラベリング ---
label_pass pass1

# --- Phase B: 少数アーキ(11対象のうち薄いもの)を狙い撃ちで追加取得 → 再ラベリング ---
if [ "${SKIP_MINORITY:-0}" != "1" ]; then
  echo "=== minority top-up fetch $(date -u +%FT%TZ) ==="
  python -u fetch_minority_archetype_episodes.py --deck-labels "$LABELS" \
    --min-decks "${MIN_DECKS:-200}" --max-episodes "${MINORITY_MAX:-600}" --workers 6 \
    --out-dir "$REPLAYS" --master-index-path "$MASTER" 2>&1 | tee "$LOG_DIR/fetch_minority.log"
  label_pass pass2
fi

# --- Phase B2: ラベル品質の監査(定義老化・他アーキの看板でのゲート通過を検出) ---
echo "=== label audit $(date -u +%FT%TZ) ==="
python deck_predictor/audit_archetype_labels.py --deck-db "$DECK_DB" --deck-labels "$LABELS" \
  --rough-predictor-json deck_predictor/rough_predictor_g2.json \
  --out "$LOG_DIR/_g2_label_audit.md" 2>&1 | tee "$LOG_DIR/audit.log"

# --- Phase C: 11アーキタイプの模倣学習(1件失敗しても続行) ---
SUMMARY="$LOG_DIR/_g2_summary.txt"
: > "$SUMMARY"
for arch in "${ARCHS[@]}"; do
  echo "=== BC pipeline: $arch $(date -u +%FT%TZ) ==="
  python policy_net/run_archetype_pipeline.py --archetype "$arch" --tag g2 \
    --replays-dir "$REPLAYS" --master-index-path "$MASTER" --deck-labels-path "$LABELS" \
    > "$LOG_DIR/pipeline_${arch}_g2.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "$arch OK" | tee -a "$SUMMARY"
  else echo "$arch FAILED (exit $rc)" | tee -a "$SUMMARY"; fi
done

echo "=== ALL DONE $(date -u +%FT%TZ) ==="
cat "$SUMMARY"
