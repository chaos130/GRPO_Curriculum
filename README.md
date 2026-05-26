# GRPO_Curriculum

本仓库当前定位为 **GRPO Curriculum 框架层**：以 **EasyR1/verl** 作为 GRPO 训练后端，以 **Mind2Web** 作为离线 Web agent benchmark/dataset 后端，并保留已有 Android GUI GRPO 实验代码。

```
GRPO_Curriculum/
├── data/            # 框架层 dataset adapters
├── prompts/         # 框架层 prompt/state builders
├── rollout/         # 框架层 rollout adapters
├── rewards/         # 框架层 reward helpers / smoke rewards
├── configs/         # 框架层实验配置
├── EasyR1/          # GRPO/verl 训练后端 + Android GUI 定制
└── Mind2Web/        # Mind2Web 原始数据集与基线代码（见官方文档）
```

---

## Mind2Web Trajectory GRPO 框架

新增代码实现了 **Mind2Web offline trajectory-level GRPO** 的第一版框架。它不是把 Mind2Web 离线转换成 EasyR1 现有 step 数据格式，而是保留 Mind2Web 原始 task/trajectory 结构：

```text
task = (S1, A1*, S2, A2*, ..., St, At*)
```

其中每一步状态 `S_i` 定义为输入给 policy LLM 的提示词：

```text
S_i = tree_repr_i + seq_input_i
```

- `tree_repr_i`：由 `cleaned_html_i` 按 `candidate_ids_i` 裁剪 DOM 后得到。
- `seq_input_i`：由 `confirmed_task` 和 `previous_actions` 构造，沿用 Mind2Web 原生 SFT prompt 文字。
- 状态序列 `S1...St` 来自 Mind2Web 离线数据，是固定的；rollout 时采样的是动作序列。

### 1. 新增框架层模块

| 文件 | 作用 |
|------|------|
| `prompts/mind2web.py` | 复刻并拆分 Mind2Web 的状态构造逻辑：`tree_repr`、`seq_input`、`state_prompt`、`seq_target` |
| `data/adapters/mind2web_trajectory.py` | task-level Dataset adapter；一个样本是一条 Mind2Web task trajectory，而不是单个 step |
| `rollout/mind2web_trajectory_rollout.py` | offline trajectory rollout；对同一 task 固定状态序列采样 `rollout.n` 条动作轨迹 |
| `rewards/mind2web_trajectory.py` | smoke-test reward，仅检查 step action 是否可解析；正式 trajectory reward 后续实现 |
| `configs/mind2web_trajectory_grpo.yaml` | Mind2Web trajectory GRPO 的最小示例配置 |

### 2. EasyR1 后端接入点

| 文件 | 改动 |
|------|------|
| `EasyR1/verl/trainer/config.py` | 新增 `data.dataset_type`、`data.rollout_type`、`data.dataset_kwargs` |
| `EasyR1/verl/trainer/data_loader.py` | 新增 dataset factory；默认仍走 `RLHFDataset`，`mind2web_trajectory` 时走新 adapter |
| `EasyR1/verl/trainer/ray_trainer.py` | 新增 `rollout_type == mind2web_trajectory` 分支；默认 rollout 路径保持不变 |

默认配置仍是：

```yaml
data:
  dataset_type: rlhf
  rollout_type: default
```

因此已有 EasyR1 / Android GUI 训练脚本不受影响。

### 3. Mind2Web Dataset 输出契约

`Mind2WebTrajectoryDataset` 保持 EasyR1 batch contract，返回：

```text
input_ids
attention_mask
position_ids
raw_prompt_ids
ground_truth
trajectory_data
```

其中 `input_ids/raw_prompt_ids` 是 task-level seed prompt，用于兼容 EasyR1 数据接口；真正用于 policy rollout 的每一步状态在：

```text
trajectory_data["steps"][i]["state_prompt"]
```

每个 step 保留：

```text
step_index
action_uid
candidate_ids
tree_repr
seq_input
state_prompt
choices
pos_candidates / neg_candidates
pos_ids
operation
target_action
seq_target
valid_positive
```

### 4. Rollout 过程

Mind2Web trajectory rollout 的单位是一个 task，而不是 step：

```text
同一个 task
  固定状态序列 S1...St
  采样 rollout.n 条动作轨迹
```

实现上 `rollout/mind2web_trajectory_rollout.py` 会：

1. 对 batch 中每个 task 创建 `rollout.n` 个 trajectory context。
2. 对第 `i` 个固定状态 `S_i` 调用 EasyR1 现有 `actor_rollout_ref_wg.generate_sequences`。
3. 收集每条轨迹的 step responses。
4. 将结果展开为 EasyR1 可继续训练的 step-action rows，并附带：
   - `uid`
   - `trajectory_id`
   - `rollout_index`
   - `step_index`
   - `step_data`
   - `trajectory_data`
   - `predicted_trajectory`

后续 reward 可以基于 `trajectory_id` 聚合整条轨迹得分，再回填到同一条轨迹的 step action 上。

### 5. 示例配置

配置文件：

```text
configs/mind2web_trajectory_grpo.yaml
```

关键字段：

```yaml
data:
  dataset_type: mind2web_trajectory
  rollout_type: mind2web_trajectory
  train_files: data/train/*.json
  val_files: data/test_task/*.json
  dataset_kwargs:
    data_path: /Users/chaos/workplace/data/Mind2Web
    candidate_source: ranked
    score_file: /Users/chaos/workplace/data/Mind2Web/src/scores_all_data.pkl
    top_k: 50
    max_candidates: 20
    previous_k: 5
    keep_html_brackets: false
    task_filter: none

worker:
  rollout:
    n: 2
```

当前 `rewards/mind2web_trajectory.py` 只是 smoke-test reward，不代表最终训练目标。下一步应实现真正的 trajectory-level reward。

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

```bash
cd EasyR1 && pip install -e . && bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

## 未纳入版本库的大文件

- `EasyR1/checkpoints/`、`EasyR1/wandb/`
- `Mind2Web/data/`（约 12GB+ JSON，若使用 Mind2Web 需自行下载）

克隆后请在本机放置数据与模型权重，并修改各脚本中的路径/API 配置。
