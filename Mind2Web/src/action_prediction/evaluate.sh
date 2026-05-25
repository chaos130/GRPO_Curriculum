#! /bin/bash
OUTPUT_DIR=./output
LOG_FILE=${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log
mkdir -p ${OUTPUT_DIR}
export HF_ENDPOINT=https://hf-mirror.com

echo "日志保存至: ${LOG_FILE}"

python evaluate_llm.py\
  +output_path=${OUTPUT_DIR}\
  +llm_prompt=llm_prompt.json\
  +policy_rate_limit=40\
  +top_k=50 2>&1 | tee ${LOG_FILE}