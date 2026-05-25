#!/bin/bash
# Debug 用：单 GPU + batch=1，跑 1 个训练 step 就停。
# 目的：验证新的 <think>/<answer> prompt，并把 rollout 轨迹 dump 成 JSON 查看。
# 不做长训、不写 wandb、不存 checkpoint。
export JUDGE_ENABLED=true
export JUDGE_API_KEY=sk-KxRYR2ovtMGU7b6jDGpYDID0evUTyUkPL6nwJKHxh5PYQ8Zk
export JUDGE_BASE_URL=https://yunwu.ai/v1   # 或 DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1
export JUDGE_MODEL=gpt-4o-mini
set -e

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

# 只使用 1 张卡，避免 global_batch_size=1 在多卡下 per-device=0 的报错
export CUDA_VISIBLE_DEVICES=0

# ========== LLM-as-Judge ==========
# 默认关闭，reward 的 reason 项会用中性 0.5 代替；联调好打分骨架后再开。
# 开启示例：
#   export JUDGE_ENABLED=true
#   export JUDGE_API_KEY=sk-xxx
#   export JUDGE_BASE_URL=https://api.openai.com/v1   # 或 DashScope/OpenRouter/本地 vLLM
#   export JUDGE_MODEL=gpt-4o-mini
export JUDGE_ENABLED=${JUDGE_ENABLED:-false}

# ========== 路径 ==========
MODEL_PATH=/workspace/model/Qwen/Qwen2.5-VL-3B-Instruct
DATA_DIR=/workspace/data/numbergame
TRAIN_DATA=${DATA_DIR}/train/data-00000-of-00001.arrow
VAL_DATA=${DATA_DIR}/test/data-00000-of-00001.arrow

EXPERIMENT_NAME=qwen2_5_vl_3b_android_gui_debug_rollout

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.rollout_batch_size=1 \
    data.val_batch_size=1 \
    data.format_prompt=examples/format_prompt/android_gui.jinja \
    data.seed=42 \
    data.filter_overlong_prompts=false \
    algorithm.kl_coef=4.0e-2 \
    worker.actor.global_batch_size=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.max_grad_norm=0.1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.model.lora.rank=32 \
    worker.actor.model.lora.exclude_modules='.*visual.*' \
    worker.actor.optim.lr=1.0e-5 \
    worker.actor.optim.weight_decay=1.0e-1 \
    worker.actor.optim.lr_warmup_ratio=0.0 \
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
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enforce_eager=true \
    worker.reward.reward_function=examples/reward_function/android_gui.py:compute_score \
    worker.reward.reward_function_kwargs.budget_target_low=30 \
    worker.reward.reward_function_kwargs.budget_target_high=120 \
    worker.reward.reward_function_kwargs.budget_hard_max=256 \
    worker.reward.reward_function_kwargs.weights.format=0.15 \
    worker.reward.reward_function_kwargs.weights.reason=0.25 \
    worker.reward.reward_function_kwargs.weights.budget=0.10 \
    worker.reward.reward_function_kwargs.weights.final=0.50 \
    trainer.total_epochs=1 \
    trainer.max_steps=1 \
    trainer.project_name=easy_r1_debug \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.val_after_train=false \
    trainer.save_freq=-1 \
    trainer.log_rollout_trajectory_json=true \
    trainer.rollout_trajectory_json_steps=[1]
