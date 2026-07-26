#!/bin/bash
set -o pipefail
cd /c/dev/pokemon-tcg-ai-battle/kaggle_replays/deck_predictor

echo "=== waiting for build_dataset.py to finish ==="
until grep -qE "書き出しました|Traceback" build_dataset_run.log 2>/dev/null; do
  sleep 5
done
echo "=== build_dataset.py finished ==="
tail -5 build_dataset_run.log

if grep -q "Traceback" build_dataset_run.log; then
  echo "PIPELINE_FAILED: build_dataset.py raised an exception"
  exit 1
fi

echo "=== running train.py ==="
python train.py 2>&1 | tee train_run.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "PIPELINE_FAILED: train.py failed"
  exit 1
fi

echo "=== running adjust_prior.py --deploy ==="
python adjust_prior.py --deploy 2>&1 | tee adjust_prior_run.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "PIPELINE_FAILED: adjust_prior.py failed"
  exit 1
fi

echo "=== running evaluate.py --all ==="
python evaluate.py --all 2>&1 | tee evaluate_run.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "PIPELINE_FAILED: evaluate.py failed"
  exit 1
fi

echo "=== running pytest test_ml_predictor.py ==="
cd /c/dev/pokemon-tcg-ai-battle/sample_submission
python -m pytest tests/unit/test_ml_predictor.py -v 2>&1 | tee /c/dev/pokemon-tcg-ai-battle/kaggle_replays/deck_predictor/pytest_run.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "PIPELINE_FAILED: pytest failed"
  exit 1
fi

echo "PIPELINE_SUCCESS: all steps completed"
