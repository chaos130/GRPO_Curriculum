# GUI_GRPO

本仓库以 **EasyR1** 为主：在 [EasyR1/veRL](https://github.com/hiyouga/EasyR1) 上用 **GRPO** 微调视觉语言模型，面向 **Android 截图数字游戏** 任务做了端到端重设计（prompt、稠密奖励、Judge、调试工具链）。

```
GUI_GRPO/
├── EasyR1/          # GRPO 训练 + Android GUI 定制
└── Mind2Web/        # Mind2Web 数据集与基线代码（见官方文档）
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

## 快速入口

```bash
cd EasyR1 && pip install -e . && bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

## 未纳入版本库的大文件

- `EasyR1/checkpoints/`、`EasyR1/wandb/`
- `Mind2Web/data/`（约 12GB+ JSON，若使用 Mind2Web 需自行下载）

克隆后请在本机放置数据与模型权重，并修改各脚本中的路径/API 配置。
