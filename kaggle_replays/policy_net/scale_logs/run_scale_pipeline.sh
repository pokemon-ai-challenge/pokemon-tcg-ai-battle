#!/bin/bash
# BCデータスケーリング曲線パイプライン: 残りの subsample / build_features / train / eval を
# 順番に実行する。f=0.125 seed=42 は既に手動実行済みなのでスキップする(冪等性のため、
# 出力ファイルが既に存在する条件は全ステージでスキップする設計。途中で落ちても再実行で
# 再開できる)。
set -e
set -o pipefail

REPO="/c/Users/rinnz/Documents/pokemon/pokemon-tcg-ai-battle"
PN="$REPO/kaggle_replays/policy_net"
RL="$REPO/kaggle_replays/rl"
WDIR="$REPO/sample_submission/ptcg_ai/learning"
LOGS="$PN/scale_logs"
SRC_JSONL="$REPO/kaggle_replays/training_data/policy_positions_marnie_grimmsnarl_ex.jsonl.gz"
FULL_FEATURES="$PN/features_marnie_grimmsnarl_ex.npz"

export PYTHONIOENCODING=utf-8

trap 'echo "[FATAL] pipeline failed at line $LINENO" | tee -a "$LOGS/PIPELINE_FAILED.marker"' ERR

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

FRACTIONS="0.125 0.25 0.5"
SEEDS="42 1 2"
ALLFRACTIONS="0.125 0.25 0.5 1.0"

cd "$PN"

log "=== Stage A: subsample_episodes.py ==="
for frac in $FRACTIONS; do
  for seed in $SEEDS; do
    if [ "$frac" = "0.125" ] && [ "$seed" = "42" ]; then continue; fi
    out="scale_data/policy_positions_scale_f${frac}_s${seed}.jsonl.gz"
    if [ -f "$out" ]; then log "skip subsample (exists): $out"; continue; fi
    log "subsample start: frac=$frac seed=$seed"
    python subsample_episodes.py --in "$SRC_JSONL" --out "$out" --fraction "$frac" --seed "$seed" \
      > "$LOGS/subsample_f${frac}_s${seed}.log" 2>&1
    log "subsample done: frac=$frac seed=$seed"
  done
done

log "=== Stage B: build_features.py ==="
for frac in $FRACTIONS; do
  for seed in $SEEDS; do
    if [ "$frac" = "0.125" ] && [ "$seed" = "42" ]; then continue; fi
    in="scale_data/policy_positions_scale_f${frac}_s${seed}.jsonl.gz"
    out="scale_features/features_scale_f${frac}_s${seed}.npz"
    if [ -f "$out" ]; then log "skip build_features (exists): $out"; continue; fi
    log "build_features start: frac=$frac seed=$seed"
    python build_features.py --in "$in" --out "$out" --weight-scheme concentrated \
      > "$LOGS/build_f${frac}_s${seed}.log" 2>&1
    log "build_features done: frac=$frac seed=$seed"
  done
done

log "=== Stage C: train.py ==="
for frac in $ALLFRACTIONS; do
  for seed in $SEEDS; do
    if [ "$frac" = "0.125" ] && [ "$seed" = "42" ]; then continue; fi
    if [ "$frac" = "1.0" ]; then
      feat="$FULL_FEATURES"
    else
      feat="scale_features/features_scale_f${frac}_s${seed}.npz"
    fi
    outw="$WDIR/policy_weights_scale_f${frac}_s${seed}.json"
    metrics="scale_metrics/metrics_f${frac}_s${seed}.json"
    if [ -f "$outw" ]; then log "skip train (exists): $outw"; continue; fi
    log "train start: frac=$frac seed=$seed feat=$feat"
    python train.py --features "$feat" --out-weights "$outw" --seed "$seed" --metrics-out "$metrics" \
      > "$LOGS/train_f${frac}_s${seed}.log" 2>&1
    log "train done: frac=$frac seed=$seed"
    if [ "$frac" != "1.0" ]; then
      rm -f "scale_features/features_scale_f${frac}_s${seed}.npz"
      log "removed intermediate npz: frac=$frac seed=$seed"
    fi
  done
done

log "=== ALL TRAINING DONE ==="

log "=== Stage D: eval_diagnostics.py (POOL8, 1600 games each) ==="
cd "$RL"
OPP="alakazam,crustle,marnie_grimmsnarl_ex,rocket_mewtwo_ex,omatsuri_ondo,shirona_garchomp_ex,ogerpon_teal_ex,dragapult_ex"
for frac in $ALLFRACTIONS; do
  for seed in $SEEDS; do
    weights="policy_weights_scale_f${frac}_s${seed}.json"
    outlog="scale_eval/eval_f${frac}_s${seed}.log"
    if [ -f "$outlog" ] && grep -q "全体" "$outlog" 2>/dev/null; then
      log "skip eval (exists): $outlog"
      continue
    fi
    log "eval start: frac=$frac seed=$seed"
    python eval_diagnostics.py --learner marnie_grimmsnarl_ex --learner-weights "$weights" \
      --opponents "$OPP" --games 1600 --workers 8 > "$outlog" 2>&1
    log "eval done: frac=$frac seed=$seed"
  done
done

log "=== PIPELINE ALL DONE ==="
touch "$LOGS/PIPELINE_DONE.marker"
