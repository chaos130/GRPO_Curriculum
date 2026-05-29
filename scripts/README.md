# 框架层脚本

本目录存放 **GRPO Curriculum 框架** 的运行与调试脚本，不写入后端目录：

| 后端 | 目录 | 职责 |
|------|------|------|
| EasyR1 | `EasyR1/` | verl GRPO 训练、vLLM rollout |
| Mind2Web | `Mind2Web/` | 数据集与官方基线 |

| 脚本 | 说明 |
|------|------|
| `env_defaults.sh` | 机器路径（`MIND2WEB_DATA`、`MODEL_PATH`、HF 缓存）；不含 API key |
| `smoke_mind2web_dataset.py` | CPU：加载 1 条 task，检查 `state_prompt` / `trajectory_data` |
| `mind2web_trajectory_debug_rollout.sh` | GPU：`max_steps=1` 跑通固定状态轨迹 rollout + reward |
| `mind2web_trajectory_grpo.sh` | GPU：Mind2Web trajectory GRPO 正式训练 |

环境：

```bash
cd EasyR1 && pip install -e .
pip install -r ../requirements-framework.txt   # lxml 等 Mind2Web 框架依赖
```

`mind2web_trajectory_debug_rollout.sh` 会在缺少 `lxml` 时自动 pip 安装。

## 正式训练

默认读 `configs/mind2web_trajectory_grpo.yaml` 中的 `train_files` / `val_files`（当前为单 shard baseline）。

```bash
cd /path/to/GUI_GRPO
bash scripts/mind2web_trajectory_grpo.sh
```

WandB（可选）：

```bash
export WANDB_API_KEY=...                    # 无 key 时用 LOGGER='["console"]'
export WANDB_MODE=offline                   # Docker 外网不稳时推荐
bash scripts/mind2web_trajectory_grpo.sh
```

常用 override：

```bash
# 全量 shard
TRAIN_FILES='train/*.json' VAL_FILES='test_task/*.json,test_website/*.json' \
  bash scripts/mind2web_trajectory_grpo.sh

# 仅 console
LOGGER='["console"]' bash scripts/mind2web_trajectory_grpo.sh

# policy 历史（轨迹间 seq_input 随采样动作变化）
PREVIOUS_ACTION_SOURCE=policy bash scripts/mind2web_trajectory_grpo.sh
```

Checkpoint：`EasyR1/checkpoints/grpo_curriculum/<experiment_name>/`。

路径：`env_defaults.sh` 在检测到 `/workspace/model` 与 `/workspace/data` 时使用 Docker 挂载路径，否则使用宿主机 `/mnt/sda/Xml/workplace/...`。
