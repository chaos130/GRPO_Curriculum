# GRPO_Curriculum

本仓库当前定位为 **GRPO Curriculum 框架层**：以 **EasyR1/verl** 作为 GRPO 训练后端，以 **Mind2Web** 作为离线 Web agent benchmark/dataset 后端，并保留已有 Android GUI GRPO 实验代码。

```
GRPO_Curriculum/
├── data/            # 框架层 dataset adapters
├── prompts/         # 框架层 prompt/state builders
├── rollout/         # 框架层 rollout adapters
├── rewards/         # 框架层 reward（Mind2Web step reward + 单元测试）
├── configs/         # 框架层实验配置
├── scripts/         # 框架层运行/调试脚本（不放进 EasyR1 / Mind2Web）
├── EasyR1/          # GRPO/verl 训练后端（仅后端与 Android 实验）
└── Mind2Web/        # Mind2Web 数据集与基线代码（仅后端）
```

**约定**：Mind2Web / EasyR1 当作后端使用；新实验的 shell、smoke test 等放在仓库根目录 `scripts/` 与 `configs/`，不要写入 `EasyR1/examples/` 或 `Mind2Web/`。

---

## Mind2Web Trajectory GRPO

在 Mind2Web **离线固定状态**上做多步 Web Agent 的 GRPO 训练。一个样本是一条完整 task 轨迹，rollout 时对每个固定状态 `S_i` 采样 `rollout.n` 条动作，再展开为 step 行做 reward / GRPO / actor 更新。

```text
task = (S1, A1*, S2, A2*, ..., St, At*)
S_i = tree_repr_i + seq_input_i   # prompts/mind2web.py
```

状态序列来自数据集（离线）；策略只学习在固定 `S_i` 上选动作。

### 1. 框架层模块

| 文件 | 作用 |
|------|------|
| `prompts/mind2web.py` | `POLICY_SYSTEM`、DOM 裁剪、`state_prompt` / `seq_target` 构造 |
| `data/adapters/mind2web_trajectory.py` | task-level Dataset；只返回 `trajectory_data` + `ground_truth`（不在此处 tokenize） |
| `rollout/mind2web_trajectory_rollout.py` | 固定状态多轨迹 rollout；逐步调用 vLLM，展开为 step 行 |
| `rewards/mind2web_trajectory.py` | 逐步 reward：`format`（连续结构分）+ `answer`（action / id / value） |
| `rewards/test_mind2web_trajectory.py` | reward 单元测试 |
| `configs/mind2web_trajectory_grpo.yaml` | 默认实验配置（对齐 wandb run `mind2web_trajectory_grpo_20260527_115746` 并含后续修正） |
| `scripts/mind2web_trajectory_grpo.sh` | 正式训练入口 |
| `scripts/mind2web_trajectory_debug_rollout.sh` | 仅 validation rollout 调试 |
| `scripts/smoke_mind2web_dataset.py` | CPU 检查数据与 `state_prompt` |
| `scripts/env_defaults.sh` | 机器路径（`MIND2WEB_DATA`、`MODEL_PATH` 等），不含 API key |

### 2. 训练数据流

```text
Mind2Web JSON
  → Mind2WebTrajectoryDataset（1 样本 = 1 task，含 steps[].state_prompt）
  → ray_trainer._make_batch_data（贴 task 级 uuid）
  → generate_mind2web_trajectory_batch（每步 tokenize state_prompt → vLLM 生成 action）
  → 展开为 step 行（batch 大小 = Σ steps × rollout.n）
  → reward（逐步打分）→ GRPO advantage（Per-state 分组）→ update_actor（+ KL loss）
```

**Tokenize 只发生在 rollout**：Dataset 不再构造无用的 seed `input_ids`；每步在 `rollout/mind2web_trajectory_rollout.py` 里对 `state_prompt` 编码。

### 3. Dataset 输出

`Mind2WebTrajectoryDataset.__getitem__` 返回：

| 字段 | 含义 |
|------|------|
| `trajectory_data` | 整条 task：`steps[]`（含 `state_prompt`、`seq_target`、`tree_repr` 等）、`previous_action_source` |
| `ground_truth` | JSON 字符串，整条 gold 轨迹（task 级） |

每个 `steps[i]` 保留：`step_index`、`action_uid`、`state_prompt`、`seq_target`、`tree_repr`、`choices`、`operation` 等（不含原始 `cleaned_html`，减小 Ray 序列化体积）。

`dataset_kwargs.previous_action_source`：

- `gold`（默认）：每步 prompt 用数据集标注的历史动作
- `policy`：用当前 rollout 已采样动作重建 `seq_input`（DOM 仍固定）

### 4. Rollout 与 GRPO 分组

对每个 task 创建 `rollout.n` 条轨迹 context，按 `step_index` 循环调用 `generate_sequences`（每步 `n=1`，随机 seed 避免两条 rollout 完全相同）。

展开后每行 metadata：

| 字段 | 含义 |
|------|------|
| `task_uid` | 本 batch 内该 task 的 uuid（数 `rollout_batch_size`、rollout JSON 聚合用） |
| `uid` | **GRPO 分组键** = `{task_uid}:{step_index}`（Per-state：同一步的 n 条 rollout 一组） |
| `trajectory_id` | `{task_uid}:{rollout_index}` |
| `step_index` / `action_uid` | 步序号 / 数据集动作 id |
| `step_data` | 该步 gold 与 DOM 元数据（reward 读 `seq_target`） |

GRPO advantage 使用 `uid` 做组内减均值除标准差，因此比较的是 **同一固定状态上的多条采样**，而不是整 task 所有 step 混在一起。

### 5. Reward（`mind2web_trajectory_step`）

逐步 outcome reward，写入每行 response 末 token：

```text
overall = 0.5 × format + 0.5 × answer
```

| 子项 | 说明 |
|------|------|
| `format` | 连续结构分（1.0 起按缺失字段 / 多余行线性扣分）；行首锚定正则 `^Element:` / `^Action:` / `^Value:` |
| `answer` | `0.3×action_hit + 0.4×id_hit + 0.3×value_hit`（与 gold `seq_target` 比） |

`format` 不重复惩罚 Value 行内容（由 `value_hit` 负责），避免双重扣分；连续 `format` 有助于 GRPO 在 answer 相同时仍能拉开组内方差。

WandB 指标示例：`val/id_hit_reward`、`val/format_reward`、`val/reward_score` 等。

### 6. EasyR1 后端接入（Mind2Web 相关）

| 文件 | 改动 |
|------|------|
| `verl/trainer/config.py` | `dataset_type` / `rollout_type` / `dataset_kwargs`；`worker.trajectory_rollout` |
| `verl/trainer/data_loader.py` | `mind2web_trajectory` → `Mind2WebTrajectoryDataset` |
| `verl/trainer/ray_trainer.py` | trajectory rollout 分支；`_pad_batch_for_policy_update`；`_val_sample_labels`（W&B 用 `seq_target` 作 label）；`task_uid` 计数 |
| `verl/workers/fsdp_workers.py` | `global_batch_size × rollout.n`；多卡 `per_device ≥ 1` |
| `verl/utils/logger/gen_logger.py` | W&B val 表截断长 DOM；上传失败不中断训练 |
| `verl/utils/rollout_trajectory.py` | rollout JSON 按 `task_uid` 聚合 task → trajectory → step |
| `verl/trainer/main.py` | 转发 `WANDB_API_KEY` / `WANDB_DIR` / `WANDB_PROJECT` 等到 Ray worker |

默认 `dataset_type: rlhf` / `rollout_type: default` 时，Android GUI 等原有路径不变。

### 7. 默认配置要点

见 `configs/mind2web_trajectory_grpo.yaml`：

| 项 | 默认 | 说明 |
|----|------|------|
| `train_files` / `val_files` | `train/train_0.json`、`test_task/test_task_0.json` | 可用环境变量 `TRAIN_FILES` / `VAL_FILES` 覆盖 |
| `max_prompt_length` | 4096 | 逐步 `state_prompt` 编码上限（Mind2Web DOM 较长） |
| `max_response_length` | 256 | 单步 action 生成长度 |
| `rollout_batch_size` | 1 | 每训练 step 几个 **task** |
| `worker.rollout.n` | 2 | 每 task 几条采样轨迹（GRPO 组内大小） |
| `worker.actor.global_batch_size` | 1 | yaml 值；worker 内 × `rollout.n` 用于多卡 |
| `algorithm.use_kl_loss` | true | ref 策略 KL 进 actor loss，抑制 reward hacking |
| `val_freq` / `save_freq` | 10 / 50 | 验证与存 checkpoint 频率 |
| `val_generations_to_log` | 4 | 每次 val 写入 W&B `val/generations` 的样本数 |

`max_num_batched_tokens` 需 ≥ `max_prompt_length + max_response_length`（默认 4352）。

### 8. 运行

```bash
cd EasyR1 && pip install -e .
pip install -r ../requirements-framework.txt   # lxml
cd ..

# CPU：检查数据
python scripts/smoke_mind2web_dataset.py

# GPU：仅 rollout + reward（不更新权重）
bash scripts/mind2web_trajectory_debug_rollout.sh

# 正式训练（路径见 scripts/env_defaults.sh）
export WANDB_API_KEY=...          # 默认 LOGGER 含 wandb；无 key 时用 LOGGER='["console"]'
export WANDB_MODE=offline         # 外网不稳时推荐，日志在 EasyR1/wandb/
bash scripts/mind2web_trajectory_grpo.sh
```

常用 override：

```bash
LOGGER='["console"]' bash scripts/mind2web_trajectory_grpo.sh
TRAIN_FILES='train/*.json' VAL_FILES='test_task/*.json' bash scripts/mind2web_trajectory_grpo.sh
PREVIOUS_ACTION_SOURCE=policy bash scripts/mind2web_trajectory_grpo.sh
```

Checkpoint：`EasyR1/checkpoints/grpo_curriculum/<experiment_name>/`（`find_last_checkpoint: true` 可续训）。

**W&B 查看 val 样例**：在 run 的 **Media / Tables** 里打开 `val/generations` 并选择 **step**；不要用 `runs.summary["val/generations"]` 表达式（Summary 视图常只显示行号、看不到单元格）。

更细的脚本说明见 `scripts/README.md`。

---

## EasyR1：相对上游的改动（Android GUI Task）

上游 EasyR1 主要提供通用 VLM + GRPO/DAPO 等训练能力（Geometry3K、math 等示例）。本仓库在 `EasyR1/examples/` 与 `verl/` 中针对 **Number Game（红绿灯选数）** 做了完整任务闭环，核心变化如下。

### 1. 任务与 Prompt 重设计

| 文件 | 作用 |
|------|------|
| `examples/format_prompt/android_gui.jinja` | 定义游戏规则、分步推理格式、输出结构 |
| `examples/qwen2_5_vl_3b_android_gui_grpo.sh` | 训练入口；支持 `YELLOW_DISAMBIGUATION` 开关 |

**任务**：根据截图中的红绿灯（绿/红/黄）与三个数字，输出应点击的位置 `0/1/2`。

**结构化输出**（训练与奖励共用）：

```xml
<think>
Light: GREEN|RED|YELLOW
Numbers: left=<n0>, middle=<n1>, right=<n2>
Rule: ...
Choice: position <0|1|2> because ...
</think>
<answer>X</answer>
```

**YELLOW 消歧**（`data.format_prompt_kwargs.yellow_disambiguation`）：

- `false`（默认）：黄灯 → 选「中间数字」
- `true`：黄灯 → 按**数值**排序取 median，再映射到**屏幕位置**（避免模型把「中间格子」当成「中间数值」）

对比实验示例：

```bash
YELLOW_DISAMBIGUATION=false bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
YELLOW_DISAMBIGUATION=true  EXPERIMENT_NAME=grpo_yellow_disambig bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

### 2. 稠密奖励函数（替代单一 0/1）

`examples/reward_function/android_gui.py` 将 outcome reward 拆成四维加权，供 GRPO 组内比较：

| 维度 | 权重（默认） | 说明 |
|------|-------------|------|
| `format` | 0.15 | think/answer 结构、字段完整性（连续分） |
| `reason` | 0.25 | **Image-grounded LLM-as-Judge**（见下） |
| `budget` | 0.10 | 推理 token 长度落在目标区间（连续、无平台） |
| `final` | 0.50 | `<answer>` 是否等于 `ground_truth` |

**Judge 设计要点**（与旧版「只看文本对错」不同）：

- 同一 prompt 的 `rollout.n` 条响应作为 **一个 group**，**一次 API 调用**完成组内比较
- Judge 输入：**截图 + N 条 trace**，**不提供 ground truth**
- Judge 自行从图中恢复灯色与三数，再评 6 个 axis（`light_correct`、`numbers_correct` 等）并给出 **严格 ranking** → 派生 `rank_score`，保证组内 reward 有方差（利于 GRPO）
- `reason = 0.6 × mean(axes) + 0.4 × rank_score`

环境变量（由 `verl/trainer/main.py` 转发到 Ray reward worker）：

`JUDGE_ENABLED`、`JUDGE_API_KEY`、`JUDGE_BASE_URL`、`JUDGE_MODEL`、`JUDGE_MAX_WORKERS` 等。

### 3. 训练框架层改动（`verl/`）

| 改动 | 位置 | 说明 |
|------|------|------|
| `format_prompt_kwargs` | `verl/trainer/config.py` | Jinja 模板可配置变量（如 `yellow_disambiguation`） |
| Rollout 轨迹 JSON | `verl/utils/rollout_trajectory.py`、`ray_trainer.py` | `trainer.log_rollout_trajectory_json=true` 时，在指定 step 导出 prompt、n 条 rollout、各子 reward、`judge_log` |
| Judge / Wandb 环境变量 | `verl/trainer/main.py` | 显式传入 Ray `runtime_env`，避免 worker 读不到 shell export |
| 消费级多卡 NCCL | `main.py` + 训练脚本 | `NCCL_P2P_DISABLE`、`NCCL_CUMEM_*` 等，适配无 NVLink 的 RTX 4090 等 |

调试脚本：`examples/qwen2_5_vl_3b_android_gui_debug_rollout.sh`（小 batch、只 dump step 1 轨迹）。

轨迹文件示例路径：

`checkpoints/<project>/<exp>/rollout_trajectories/step_0001.json`

### 4. 数据与真机工具链（`examples/android_gui_cookbook/`）

| 组件 | 文件 | 说明 |
|------|------|------|
| 游戏部署 | `game_docker/`、`README.md` | Docker/K8s 部署 HTML Number Game |
| 数据采集 | `collect_data.py` | 多设备 ADB 并发截图 + 元数据，用于构建 arrow 训练集 |
| 在线试玩 | `play_agent.py`、`adb_controller.py`、`vlm_client.py` | 真机 VLM Agent，验证策略 |

训练数据一般为 **截图 + `ground_truth` 位置** 的 arrow/HF 数据集（脚本内 `DATA_DIR` 可配置）。

### 5. 推荐训练命令

```bash
cd EasyR1
pip install -e .
bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

更完整的部署、采集、评测说明见 `EasyR1/examples/android_gui_cookbook/README.md`。

---

## 快速入口

**Mind2Web trajectory GRPO（推荐路径）**

```bash
cd EasyR1 && pip install -e .
pip install -r ../requirements-framework.txt
cd ..
bash scripts/mind2web_trajectory_grpo.sh
```

**Android GUI（历史路径，脚本在 EasyR1/examples/）**

```bash
cd EasyR1 && pip install -e . && bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

更完整的 Android 部署与采集见 `EasyR1/examples/android_gui_cookbook/README.md`。

## 未纳入版本库的大文件

- `EasyR1/checkpoints/`、`EasyR1/wandb/`
- `Mind2Web/data/`（约 12GB+ JSON，若使用 Mind2Web 需自行下载）

克隆后请在本机放置数据与模型权重，并修改各脚本中的路径/API 配置。
