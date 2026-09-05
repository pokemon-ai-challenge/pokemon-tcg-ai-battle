#!/usr/bin/env bash
cd "$(dirname "$0")"
while ! grep -q "書き出し完了" _build_base_ctrl.log 2>/dev/null; do sleep 5; done
sleep 2
echo "control npz ready, starting control training"
PYTHONIOENCODING=utf-8 python train.py --features features_base_ctrl.npz \
  --out-weights ../../sample_submission/ptcg_ai/learning/policy_weights_base_ctrl.json
echo "=== control training finished, exit=$? ==="
