#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "=== [1/2] build features (tier1a only) ==="
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python build_features.py --tiers tier1a --out features_tier1a.npz || { echo "BUILD FAILED"; exit 1; }
echo "=== [2/2] train tier1a ==="
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python train.py --features features_tier1a.npz \
  --out-weights ../../sample_submission/ptcg_ai/learning/policy_weights_tier1a.json || { echo "TRAIN FAILED"; exit 1; }
echo "=== tier1a pipeline done, exit=0 ==="
