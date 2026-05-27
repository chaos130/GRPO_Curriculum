# 框架层脚本

本目录存放 **GRPO Curriculum 框架** 的运行与调试脚本，不写入后端目录：

| 后端 | 目录 | 职责 |
|------|------|------|
| EasyR1 | `EasyR1/` | verl GRPO 训练、vLLM rollout |
| Mind2Web | `Mind2Web/` | 数据集与官方基线 |

| 脚本 | 说明 |
|------|------|
| `smoke_mind2web_dataset.py` | CPU：加载 1 条 task，检查 `state_prompt` / `trajectory_data` |
| `mind2web_trajectory_debug_rollout.sh` | GPU：`max_steps=1` 跑通固定状态轨迹 rollout + reward |
| `mind2web_trajectory_grpo.sh` | GPU：Mind2Web trajectory GRPO 正式训练（checkpoint / val / rollout JSON） |

环境：

```bash
cd EasyR1 && pip install -e .
pip install -r ../requirements-framework.txt   # lxml 等 Mind2Web 框架依赖
```

`mind2web_trajectory_debug_rollout.sh` 会在缺少 `lxml` 时自动 pip 安装。

训练示例（默认启用 wandb；未设置 `WANDB_API_KEY` 会直接报错）：

```bash
export WANDB_API_KEY=...   # https://wandb.ai/authorize
cd EasyR1 && pip install -e .
pip install -r ../requirements-framework.txt
bash ../scripts/mind2web_trajectory_grpo.sh
```

常用 override：

```bash
# 默认 `train_0.json` + `test_task_0.json`；全量数据：
TRAIN_FILES=/workspace/data/Mind2Web/data/train/*.json \
VAL_FILES=/workspace/data/Mind2Web/data/test_*/*.json \
bash scripts/mind2web_trajectory_grpo.sh

# 单 shard 调试：
TRAIN_FILE=/workspace/data/Mind2Web/data/train/train_0.json \
MAX_STEPS=100 \
PREVIOUS_ACTION_SOURCE=policy \
bash scripts/mind2web_trajectory_grpo.sh

# 仅 console，不用 wandb
LOGGER='["console"]' bash scripts/mind2web_trajectory_grpo.sh
```

Checkpoint 默认写入 `EasyR1/checkpoints/grpo_curriculum/<experiment_name>/`。

路径：`scripts/env_defaults.sh` 会在检测到 `/workspace/model` 与 `/workspace/data` 时使用 Docker 挂载路径，否则使用宿主机 `/mnt/sda/Xml/workplace/...`。也可手动 export：

```bash
export MODEL_PATH=/workspace/model/Qwen/Qwen2.5-VL-3B-Instruct
export MIND2WEB_DATA=/workspace/data/Mind2Web/data
```
