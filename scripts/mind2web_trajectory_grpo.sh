#!/bin/bash
# Mind2Web trajectory GRPO
#
# Tuned for this machine (auto-detected each run):
#   RAM ~128GB  |  2× NVIDIA RTX 4090 D (~24GB VRAM each)
#
#   export WANDB_API_KEY=... && bash scripts/mind2web_trajectory_grpo.sh
#
# If Docker only mounts 1 GPU, script auto-falls back to 1-GPU settings.
# Force 2-GPU: ensure container sees both cards, then e.g. N_GPUS=2 TP_SIZE=2
#
# Defaults match wandb run mind2web_trajectory_grpo_20260527_115746 (see configs/mind2web_trajectory_grpo.yaml).
# Paths (data_path, model) come from env_defaults.sh; train/val shards from yaml unless overridden below.

set -euo pipefail
trap 'echo "ERROR: failed at line ${LINENO} (exit $?)" >&2' ERR

echo "[mind2web_trajectory_grpo] $(date -Iseconds) starting..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EASYR1_ROOT="${REPO_ROOT}/EasyR1"

# shellcheck source=env_defaults.sh
source "${SCRIPT_DIR}/env_defaults.sh"
cd "${EASYR1_ROOT}"

export NCCL_P2P_DISABLE=1 \
       NCCL_IB_DISABLE=1 \
       NCCL_CUMEM_ENABLE=0 \
       NCCL_CUMEM_HOST_ENABLE=0 \
       NCCL_NVLS_ENABLE=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
       HF_HUB_ENABLE_HF_TRANSFER=0 \
       RAY_TMPDIR="${RAY_TMPDIR}" \
       TMPDIR="${TMPDIR}" \
       HF_HOME="${HF_HOME}" \
       HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
       HF_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

export WANDB_DIR="${WANDB_DIR:-${EASYR1_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo_curriculum}"
mkdir -p "$WANDB_DIR"

# --- hardware probe ---
_HOST_MEM_GB="$(awk '/MemTotal/ {print int($2/1024/1024 + 0.5)}' /proc/meminfo)"
if command -v nvidia-smi >/dev/null 2>&1; then
  _NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  # Avoid `| head -1` with pipefail (SIGPIPE exit 141).
  _GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
  _GPU_NAME="${_GPU_NAME%%$'\n'*}"
  _GPU_NAME="${_GPU_NAME#"${_GPU_NAME%%[![:space:]]*}"}"
  _GPU_VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)"
  _GPU_VRAM_MIB="${_GPU_VRAM_MIB%%$'\n'*}"
  _GPU_VRAM_MIB="${_GPU_VRAM_MIB//[[:space:]]/}"
  _GPU_VRAM_GB="$(awk -v mib="${_GPU_VRAM_MIB}" 'BEGIN { printf "%.0f", mib/1024 }')"
else
  _NGPU=0
  _GPU_NAME="none"
  _GPU_VRAM_GB=0
fi

# --- profile: 2×24GB + >=96GB RAM (this host) vs 1×24GB ---
if [[ -z "${N_GPUS:-}" ]]; then
  [[ "${_NGPU}" -ge 2 ]] && N_GPUS=2 || N_GPUS=1
fi
[[ "${N_GPUS}" -ge 1 ]] || { echo "ERROR: no GPU (nvidia-smi -L empty)." >&2; exit 1; }

: "${TP_SIZE:=${N_GPUS}}"
: "${MAX_BATCHED_TOKENS:=4352}"
: "${LIMIT_IMAGES:=1}"
: "${LR:=1.0e-6}"
: "${OPTIM_STRATEGY:=adamw_bf16}"
: "${ROLLOUT_N:=2}"
: "${TOTAL_EPOCHS:=1}"
: "${ROLLOUT_BATCH_SIZE:=1}"
: "${VAL_BATCH_SIZE:=1}"
: "${GLOBAL_BATCH_SIZE:=1}"
: "${SAVE_FREQ:=50}"
: "${VAL_FREQ:=10}"
: "${VAL_GENERATIONS:=4}"
: "${VAL_AFTER_TRAIN:=true}"
: "${ROLLOUT_JSON_STEPS:=[1,10,50,100]}"
: "${LOGGER:=[\"console\",\"wandb\"]}"
: "${EXPERIMENT_NAME:=mind2web_trajectory_grpo_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  [[ "${N_GPUS}" -ge 2 ]] && export CUDA_VISIBLE_DEVICES=0,1 || export CUDA_VISIBLE_DEVICES=0
fi

_TORCH_NGPU="$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)"
if [[ "${_TORCH_NGPU}" =~ ^[0-9]+$ ]] && [[ "${_TORCH_NGPU}" -ge 1 ]] && [[ "${_TORCH_NGPU}" -lt "${_NGPU}" ]]; then
  _NGPU="${_TORCH_NGPU}"
fi
if [[ "${_TORCH_NGPU}" =~ ^[0-9]+$ ]] && [[ "${_TORCH_NGPU}" -lt "${N_GPUS}" ]]; then
  echo "ERROR: N_GPUS=${N_GPUS} but torch sees ${_TORCH_NGPU} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})" >&2
  nvidia-smi -L >&2 || true
  echo "Recreate container with: docker run --gpus all ..." >&2
  exit 1
fi

: "${LORA_RANK:=32}"
EXTRA_ARGS=(
  "worker.actor.model.enable_gradient_checkpointing=true"
  "worker.actor.model.lora.rank=${LORA_RANK}"
  "worker.actor.model.lora.exclude_modules=.*visual.*"
)
if [[ "${N_GPUS}" -ge 2 && "${_HOST_MEM_GB}" -ge 96 ]]; then
  # 2×4090 24GB: LoRA + vLLM TP=2; keep headroom for update_actor after vLLM sleep
  : "${ENABLE_KL:=${ENABLE_KL:-1}}"
  : "${GPU_MEM_UTIL:=${GPU_MEM_UTIL:-0.28}}"
  EXTRA_ARGS+=(
    "worker.actor.fsdp.enable_cpu_offload=false"
    "worker.actor.model.freeze_vision_tower=true"
    "worker.rollout.enforce_eager=false"
    "worker.ref.fsdp.enable_cpu_offload=true"
  )
else
  # 1×24GB or low host RAM: actor+vLLM share one card
  : "${ENABLE_KL:=${ENABLE_KL:-1}}"
  : "${GPU_MEM_UTIL:=${GPU_MEM_UTIL:-0.25}}"
  EXTRA_ARGS+=(
    "worker.actor.fsdp.enable_cpu_offload=true"
    "worker.actor.model.freeze_vision_tower=true"
    "worker.rollout.enforce_eager=true"
  )
  export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
fi

[[ "${TP_SIZE}" == "${N_GPUS}" ]] || { echo "ERROR: TP_SIZE must equal N_GPUS" >&2; exit 1; }

command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true

echo "=== hardware ==="
echo "RAM: ${_HOST_MEM_GB} GiB | GPUs (nvidia-smi): ${_NGPU} | torch: ${_TORCH_NGPU}"
echo "GPU: ${_GPU_NAME} ~${_GPU_VRAM_GB} GiB VRAM"
echo "=== training ==="
echo "N_GPUS=${N_GPUS} TP=${TP_SIZE} KL=${ENABLE_KL} LoRA=${LORA_RANK} vllm_mem=${GPU_MEM_UTIL} CUDA=${CUDA_VISIBLE_DEVICES}"
echo "DATA train=${TRAIN_FILES:-<yaml>} val=${VAL_FILES:-<yaml>}"
echo "BATCH rollout_bs=${ROLLOUT_BATCH_SIZE} val_bs=${VAL_BATCH_SIZE} global_bs=${GLOBAL_BATCH_SIZE} rollout_n=${ROLLOUT_N} epochs=${TOTAL_EPOCHS}"
echo "MODEL=${MODEL_PATH}"

[[ -n "${MAX_STEPS:-}" ]] && EXTRA_ARGS+=("trainer.max_steps=${MAX_STEPS}")

if ! python3 -c "import lxml.etree" 2>/dev/null; then
  echo "[mind2web_trajectory_grpo] installing lxml..."
  python3 -m pip install -q -r "${REPO_ROOT}/requirements-framework.txt"
fi

echo "[mind2web_trajectory_grpo] launching trainer (logs may pause during Ray/vLLM init)..."
CLI_ARGS=(
    "config=${REPO_ROOT}/configs/mind2web_trajectory_grpo.yaml"
    "data.dataset_kwargs.data_path=${MIND2WEB_DATA}"
    "data.dataset_kwargs.score_file=${SCORE_FILE}"
    "data.dataset_kwargs.previous_action_source=${PREVIOUS_ACTION_SOURCE:-gold}"
    "worker.actor.model.model_path=${MODEL_PATH}"
    "worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
    "worker.actor.optim.lr=${LR}"
    "worker.actor.optim.strategy=${OPTIM_STRATEGY}"
    "worker.rollout.n=${ROLLOUT_N}"
    "worker.rollout.tensor_parallel_size=${TP_SIZE}"
    "worker.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
    "worker.rollout.limit_images=${LIMIT_IMAGES}"
    "worker.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS}"
    "worker.reward.reward_function=${REPO_ROOT}/rewards/mind2web_trajectory.py:compute_score"
    "algorithm.disable_kl=$([[ "${ENABLE_KL}" == 1 ]] && echo false || echo true)"
    "algorithm.use_kl_loss=$([[ "${ENABLE_KL}" == 1 ]] && echo true || echo false)"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.logger=${LOGGER}"
    "trainer.n_gpus_per_node=${N_GPUS}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.val_freq=${VAL_FREQ}"
    "trainer.val_generations_to_log=${VAL_GENERATIONS}"
    "trainer.val_after_train=${VAL_AFTER_TRAIN}"
    "trainer.rollout_trajectory_json_steps=${ROLLOUT_JSON_STEPS}"
)
# Optional overrides (default: use yaml train_files / val_files)
[[ -n "${TRAIN_FILES:-}" ]] && CLI_ARGS+=("data.train_files=${TRAIN_FILES}")
[[ -n "${VAL_FILES:-}" ]] && CLI_ARGS+=("data.val_files=${VAL_FILES}")
[[ -n "${ROLLOUT_BATCH_SIZE:-}" ]] && CLI_ARGS+=("data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}")
[[ -n "${VAL_BATCH_SIZE:-}" ]] && CLI_ARGS+=("data.val_batch_size=${VAL_BATCH_SIZE}")

exec python3 -u -m verl.trainer.main "${CLI_ARGS[@]}" "${EXTRA_ARGS[@]}"
