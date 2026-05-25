#!/bin/bash
set -euo pipefail

OUTPUT_DIR=./output_refine
LOG_FILE=${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log
mkdir -p "${OUTPUT_DIR}"
export HF_ENDPOINT=https://hf-mirror.com

echo "日志保存至: ${LOG_FILE}"

# Run as a module so that `refine.evaluator` / `refine.prompts` relative
# imports resolve. CWD must be src/action_prediction.
cd "$(dirname "$0")"
PYTHONPATH="$(pwd):$(pwd)/..:${PYTHONPATH:-}" \
python -m refine.run_refine \
  output_path="${OUTPUT_DIR}" \
  policy_llm=gpt-3.5-turbo \
  judge_llm=gpt-3.5-turbo \
  policy_rate_limit=60 \
  judge_rate_limit=60 \
  refine.max_rounds=4 \
  refine.score_threshold=4.0 \
  refine.top_k=50 \
  refine.max_candidates=20 \
  limit=3 \
  2>&1 | tee "${LOG_FILE}"
