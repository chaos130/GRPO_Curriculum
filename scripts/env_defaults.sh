# Shared path defaults for framework scripts (host vs Docker).
# Source from bash:  source "$(dirname "$0")/env_defaults.sh"
#
# Override any variable before sourcing, or export MODEL_PATH=... before running scripts.

if [[ -d /workspace/model && -d /workspace/data ]]; then
  # Docker: -v .../model:/workspace/model -v .../data:/workspace/data
  : "${MODEL_PATH:=/workspace/model/Qwen/Qwen2.5-VL-3B-Instruct}"
  : "${MIND2WEB_DATA:=/workspace/data/Mind2Web/data}"
  : "${SCORE_FILE:=/workspace/data/Mind2Web/src/scores_all_data.pkl}"
  : "${RAY_TMPDIR:=/mnt/sda/ray_tmp}"
else
  # Host
  : "${MODEL_PATH:=/mnt/sda/Xml/workplace/model/Qwen/Qwen2.5-VL-3B-Instruct}"
  : "${MIND2WEB_DATA:=/mnt/sda/Xml/workplace/data/Mind2Web/data}"
  : "${SCORE_FILE:=/mnt/sda/Xml/workplace/data/Mind2Web/src/scores_all_data.pkl}"
  : "${RAY_TMPDIR:=/mnt/sda/ray_tmp}"
fi

# Keep Hugging Face / datasets cache and temporary files off the container root
# filesystem, which is often much smaller than the mounted data disk.
: "${TMPDIR:=${RAY_TMPDIR}/tmp}"
: "${HF_HOME:=${RAY_TMPDIR}/huggingface}"
: "${HF_DATASETS_CACHE:=${HF_HOME}/datasets}"
: "${HF_HUB_CACHE:=${HF_HOME}/hub}"

# Optional: export TRAIN_FILES / VAL_FILES to override yaml (e.g. train/*.json).
# Default train/val shards are in configs/mind2web_trajectory_grpo.yaml (115746 baseline).
