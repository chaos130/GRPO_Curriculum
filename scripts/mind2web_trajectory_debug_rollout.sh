#!/bin/bash
# Mind2Web trajectory GRPO — rollout-only debug (framework script).
#
# Backends: EasyR1 (verl train/rollout) + Mind2Web (data via dataset_kwargs).
# Runs validation once (trainer.val_only=true): fixed-state trajectory rollout
# + smoke reward, no policy update. Needs 1 GPU + vLLM.
#
# Usage (from repo root):
#   bash scripts/mind2web_trajectory_debug_rollout.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EASYR1_ROOT="${REPO_ROOT}/EasyR1"

# Host vs Docker path defaults (override via env before running)
# shellcheck source=env_defaults.sh
source "${SCRIPT_DIR}/env_defaults.sh"

cd "${EASYR1_ROOT}"

# ========== env ==========
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HUB_ENABLE_HF_TRANSFER=0
export RAY_TMPDIR
export TMPDIR
export HF_HOME
export HF_DATASETS_CACHE
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${N_GPUS:-2}"
TP_SIZE="${TP_SIZE:-2}"

REWARD_FN="${REPO_ROOT}/rewards/mind2web_trajectory.py:compute_score"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mind2web_trajectory_debug_rollout}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  echo "  Docker: mount -v .../model:/workspace/model and ensure Qwen2.5-VL-3B-Instruct is present" >&2
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "ERROR: VAL_FILE does not exist: ${VAL_FILE}" >&2
  exit 1
fi

# Mind2Web DOM pruning (prompts/mind2web.py -> data_utils/dom_utils)
if ! python3 -c "import lxml.etree" 2>/dev/null; then
  echo "Installing framework deps from ${REPO_ROOT}/requirements-framework.txt ..."
  python3 -m pip install -q -r "${REPO_ROOT}/requirements-framework.txt"
fi

echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using MIND2WEB_DATA=${MIND2WEB_DATA}"
echo "Using VAL_FILE=${VAL_FILE}"
echo "Using TMPDIR=${TMPDIR}"
echo "Using HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

python3 -m verl.trainer.main \
    data.dataset_type=mind2web_trajectory \
    data.rollout_type=mind2web_trajectory \
    data.train_files="${VAL_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    data.rollout_batch_size=1 \
    data.val_batch_size=1 \
    data.shuffle=false \
    data.seed=42 \
    data.filter_overlong_prompts=false \
    data.dataset_kwargs.data_path="${MIND2WEB_DATA}" \
    data.dataset_kwargs.candidate_source=ranked \
    data.dataset_kwargs.score_file="${SCORE_FILE}" \
    data.dataset_kwargs.top_k=50 \
    data.dataset_kwargs.max_candidates=20 \
    data.dataset_kwargs.previous_k=5 \
    data.dataset_kwargs.keep_html_brackets=false \
    data.dataset_kwargs.task_filter=none \
    data.dataset_kwargs.previous_action_source="${PREVIOUS_ACTION_SOURCE:-gold}" \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=false \
    algorithm.kl_coef=1.0e-2 \
    algorithm.use_kl_loss=true \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.trust_remote_code=true \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.ref.fsdp.torch_dtype=bf16 \
    worker.ref.fsdp.enable_cpu_offload=false \
    worker.ref.offload.offload_params=true \
    worker.rollout.n=2 \
    worker.rollout.temperature=0.8 \
    worker.rollout.top_p=0.95 \
    worker.rollout.limit_images=0 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.tensor_parallel_size=${TP_SIZE} \
    worker.rollout.enforce_eager=false \
    "worker.reward.reward_function=${REWARD_FN}" \
    trainer.project_name=grpo_curriculum \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.val_after_train=false \
    trainer.val_freq=-1 \
    trainer.total_epochs=1 \
    trainer.max_steps=1 \
    trainer.save_freq=-1 \
    trainer.val_generations_to_log=4 \
    trainer.log_rollout_trajectory_json=true \
    trainer.rollout_trajectory_json_steps=[1]
