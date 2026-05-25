# GUI_GRPO

本仓库汇总 **GUI Agent** 相关的两条研究线：

- **EasyR1**：在 [EasyR1/veRL](https://github.com/hiyouga/EasyR1) 上，用 **GRPO** 微调视觉语言模型，面向 **Android 截图数字游戏** 任务做了端到端重设计（prompt、稠密奖励、Judge、调试工具链）。
- **Mind2Web**：在 [OSU Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) 基线上，为 **网页动作预测** 增加了 **Self-Refine**（Policy + Judge 多轮迭代）评测管线。

```
GUI_GRPO/
├── EasyR1/          # GRPO 训练 + Android GUI 定制
└── Mind2Web/        # Mind2Web 数据 + Self-Refine 评测
```

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

## Mind2Web：Self-Refine 改动说明

上游 Mind2Web 提供候选生成 + **单次**动作预测（`action_prediction/train.py`、`evaluate.py`）。本仓库在 `src/action_prediction/refine/` 新增 **Self-Refine** 管线：每一步动作由 **Policy LLM** 生成，**Judge LLM** 打分并反馈，多轮迭代直到达标或达到 `max_rounds`。

### 1. 与原版 evaluate 的差异

| | 原版 `evaluate.py` / `evaluate_llm.py` | 本仓库 `refine/` |
|--|----------------------------------------|------------------|
| 每步调用 | Policy 一次 → 直接算 metric | Policy ↔ Judge 多轮 |
| 失败处理 | 无迭代 | Judge `feedback` 写入 `refinement_history`，下一轮 Policy 据此修正 |
| 早停 | — | `total_score >= score_threshold` 则停止 |
| 最终预测 | 单次输出 | **最后一轮** Policy 输出（`ref_history[-1]`） |

### 2. Self-Refine 循环（单步）

```
固定输入: task, pruned HTML + candidates, trajectory_history
refinement_history = []

for round = 1 .. max_rounds:
    Policy(task, html, history, refinement_history, previous_feedback)
        → Element / Action / Value
    Judge(task, html, round, candidate_action)
        → 4 维分数 + feedback
    append to refinement_history
    if total_score >= threshold: break

final_prediction = 最后一轮 Policy 输出
→ 与 GT 比较 element_acc / action_f1 / step_acc
```

实现见 `src/action_prediction/refine/evaluator.py` 中 `RefinementActionEvaluator._refine_one_step`。

### 3. Judge 评分维度（总分 0–7）

定义于 `refine/prompts.py`（`JUDGE_SYSTEM`）与 `refine/judge.py`：

| 维度 | 范围 | 含义 |
|------|------|------|
| `action_validity` | 0–2 | 元素是否在候选列表、动作类型是否匹配 |
| `semantic_alignment` | 0–2 | 动作是否服务于用户任务 |
| `progress_contribution` | 0–2 | 是否实质性推进任务 |
| `redundancy_loop_detection` | 0–1 | 是否重复此前失败尝试 |

`total = sum(四维)`；默认 `score_threshold=5.0`（`conf/refine.yaml` 可调）。

### 4. Policy 侧设计

- **Round 1**：`policy_temperature=0`（确定性）
- **Round ≥2**：温度按 `policy_temperature_boost` 递增，鼓励探索不同元素/动作
- Prompt 显式要求：根据 Judge feedback **换元素、改动作类型、改 value**，且不要重复负反馈动作

### 5. 配置与运行

Hydra 配置：`src/action_prediction/conf/refine.yaml`

- `policy_llm` / `judge_llm`：可配不同模型（如 Qwen3-32B + Qwen2.5-72B）
- 多 API 提供商：`evaluate_llm.OpenaiEngine`（含 **Qwen3 `enable_thinking=False`** 兼容）
- `refine.max_rounds`、`top_k`、`max_candidates`、`num_workers` 等

```bash
cd Mind2Web/src/action_prediction
bash refine.sh
# 或
python -m refine.run_refine --config-path ../conf --config-name refine
```

输出目录（默认）：`output_refine/` — 含 `*_predictions_refine.json`、`*_trajectories_refine.json`（含每轮 policy/judge 原文与分数）。

### 6. 数据说明

- 训练/测试 JSON 需自行按 [Mind2Web 官方](https://huggingface.co/datasets/osunlp/Mind2Web) 准备；`data/` **不纳入 Git**（体积过大）
- `conf/refine.yaml` 中 `data.data_path`、`data.score_file` 请改为你本机路径

---

## 两条线的关系

| 项目 | 场景 | 优化方式 | 核心创新点 |
|------|------|----------|------------|
| **EasyR1** | Android 截图、离散选点 | **GRPO** 在线 RL | 稠密奖励 + 看图 group Judge + rollout 轨迹调试 |
| **Mind2Web** | 网页 DOM、元素+动作 | **离线 Self-Refine** 推理 | Policy–Judge 多轮迭代 + 结构化反馈 |

二者互补：Mind2Web 探索「Judge 引导的多轮决策」；EasyR1 将类似思想融入 RL 奖励（组内 Judge ranking），并在真机 GUI 游戏上端到端训练。

---

## 快速入口

```bash
# EasyR1 — GRPO 训练
cd EasyR1 && pip install -e . && bash examples/qwen2_5_vl_3b_android_gui_grpo.sh

# Mind2Web — Self-Refine 评测（需先准备 data/）
cd Mind2Web/src/action_prediction && bash refine.sh
```

## 未纳入版本库的大文件

- `EasyR1/checkpoints/`、`EasyR1/wandb/`
- `Mind2Web/data/`（约 12GB+ JSON）

克隆后请在本机放置数据与模型权重，并修改各脚本中的路径/API 配置。
