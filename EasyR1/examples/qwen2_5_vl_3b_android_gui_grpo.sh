#!/bin/bash

set -e

# Run from repo root so config=examples/... paths resolve correctly
cd "$(dirname "$0")/.."

# ========== 环境变量 ==========
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export RAY_TMPDIR=/mnt/sda/ray_tmp
mkdir -p "$RAY_TMPDIR"

# ========== wandb ==========
# 在 https://wandb.ai/authorize 获取；也可在运行前 export WANDB_API_KEY=...
export WANDB_API_KEY=${WANDB_API_KEY:-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}
export WANDB_DIR="$(pwd)/wandb"
mkdir -p "$WANDB_DIR"

# ========== LLM-as-Judge ==========
# 见 examples/reward_function/android_gui.py 顶部说明。Judge 必须是 VLM。
# 训练前务必：(1) 先跑 1 步抽样检查 step JSON 里 judge_log.parsed.image_observation
#   和 ground_truth 一致率，确认 judge 看图准；(2) 否则降低 weights.reason 或换更强的 judge 模型。
export JUDGE_ENABLED=${JUDGE_ENABLED:-true}
export JUDGE_API_KEY=${JUDGE_API_KEY:-sk-KxRYR2ovtMGU7b6jDGpYDID0evUTyUkPL6nwJKHxh5PYQ8Zk}
export JUDGE_BASE_URL=${JUDGE_BASE_URL:-https://yunwu.ai/v1}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o-mini}
export JUDGE_MAX_WORKERS=${JUDGE_MAX_WORKERS:-8}

# ========== Prompt 模板参数 ==========
# YELLOW 消歧：true=在 prompt 里明确「按数值中间 rank 选，不是中间格子」；false=旧版简短规则
# 对比实验示例：
#   YELLOW_DISAMBIGUATION=false bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
#   YELLOW_DISAMBIGUATION=true  EXPERIMENT_NAME=grpo_yellow_disambig bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
export YELLOW_DISAMBIGUATION=${YELLOW_DISAMBIGUATION:-false}

# ========== 路径 ==========
MODEL_PATH=/workspace/model/Qwen/Qwen2.5-VL-3B-Instruct
DATA_DIR=/workspace/data/numbergame
TRAIN_DATA=${DATA_DIR}/train/data-00000-of-00001.arrow
VAL_DATA=${DATA_DIR}/test/data-00000-of-00001.arrow

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_3b_android_gui_grpo_$(date +%Y%m%d_%H%M%S)}

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.rollout_batch_size=16 \
    data.val_batch_size=60 \
    data.format_prompt=examples/format_prompt/android_gui.jinja \
    data.format_prompt_kwargs.yellow_disambiguation=${YELLOW_DISAMBIGUATION} \
    data.seed=42 \
    data.filter_overlong_prompts=false \
    algorithm.kl_coef=4.0e-2 \
    worker.actor.global_batch_size=16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.max_grad_norm=0.1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.model.lora.rank=32 \
    worker.actor.model.lora.exclude_modules='.*visual.*' \
    worker.actor.optim.lr=1.0e-5 \
    worker.actor.optim.weight_decay=1.0e-1 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.optim.lr_scheduler_type=constant \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.ref.offload.offload_params=true \
    worker.rollout.n=3 \
    worker.rollout.temperature=0.9 \
    worker.rollout.top_p=0.95 \
    worker.rollout.limit_images=1 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.enforce_eager=true \
    worker.reward.reward_function=examples/reward_function/android_gui.py:compute_score \
    worker.reward.reward_function_kwargs.budget_target_low=30 \
    worker.reward.reward_function_kwargs.budget_target_high=120 \
    worker.reward.reward_function_kwargs.budget_hard_max=256 \
    worker.reward.reward_function_kwargs.weights.format=0.15 \
    worker.reward.reward_function_kwargs.weights.reason=0.25 \
    worker.reward.reward_function_kwargs.weights.budget=0.10 \
    worker.reward.reward_function_kwargs.weights.final=0.50 \
    trainer.total_epochs=3 \
    trainer.project_name=easy_r1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_freq=10 \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=10 \
    trainer.val_before_train=false \
    trainer.val_after_train=true \
    trainer.log_rollout_trajectory_json=true \
    trainer.rollout_trajectory_json_steps=[1,2,5,10,50,100,200]
