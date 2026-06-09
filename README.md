# GUI_GRPO

本仓库基于 [EasyR1/veRL](https://github.com/hiyouga/EasyR1)，使用 GRPO 微调视觉语言模型。目前已经打通 **Android 截图 Number Game** 的端到端训练闭环，并在 `Xml` 分支加入基于 Beta 后验的目标感知 Thompson Sampling，用于研究模型能力与任务难度匹配。

```text
GUI_GRPO/
├── EasyR1/          # GRPO 训练、Android GUI 定制与课程采样
└── Mind2Web/        # Mind2Web 数据集与基线代码
```

## Android GUI GRPO

Number Game 要求模型观察截图中的交通灯颜色和三个数字，根据规则输出应点击的位置 `0/1/2`。训练提示词要求模型生成结构化推理与答案：

```xml
<think>
Light: GREEN|RED|YELLOW
Numbers: left=<n0>, middle=<n1>, right=<n2>
Rule: ...
Choice: position <0|1|2> because ...
</think>
<answer>X</answer>
```

核心文件如下：

| 文件 | 作用 |
|---|---|
| `EasyR1/examples/format_prompt/android_gui.jinja` | 游戏规则、推理格式与 Yellow 消歧 |
| `EasyR1/examples/reward_function/android_gui.py` | Android GUI 稠密奖励函数 |
| `EasyR1/examples/qwen2_5_vl_3b_android_gui_grpo.sh` | Qwen2.5-VL-3B GRPO 训练入口 |
| `EasyR1/examples/qwen2_5_vl_3b_android_gui_debug_rollout.sh` | 小 batch rollout 调试入口 |
| `EasyR1/examples/android_gui_cookbook/` | 游戏部署、数据采集与真机 Agent 工具链 |

奖励函数由四部分组成：`format=0.15`、`reason=0.25`、`budget=0.10`、`final=0.50`。其中 `reason` 使用 image-grounded group-wise LLM-as-Judge：Judge 同时观察截图与同一道题的多条 rollout，并进行多维评分和严格排序。该排序可以为 GRPO 提供组内 reward 差异，但它与任务本身的自然成功概率是两个不同信号。

训练框架支持将 prompt、同组 rollout、各项 reward、advantage 与 Judge trace 导出到：

```text
checkpoints/<project>/<experiment>/rollout_trajectories/step_<N>.json
```

## Beta 后验与目标感知 Thompson Sampling

普通 Thompson Sampling 会优先选择成功率最高的题，容易持续采样简单任务。本分支实现的是 **target-aware Thompson Sampling**：为每道题维护二值成功率的 Beta 后验，并优先选择后验采样值接近目标成功率的题。

对于任务 `i`，每次 rollout 得到成功或失败后更新：

```text
p_i ~ Beta(alpha_i, beta_i)
alpha_i <- alpha_i + successes_i
beta_i  <- beta_i  + failures_i
```

采样器默认偏好 `p_i ≈ 0.5`，同时混入均匀探索，避免早期估计噪声导致部分任务永久失去采样机会。Beta 后验只使用最终答案是否正确进行更新，不使用连续 `overall reward` 或 Judge 排名，因此后验仍可解释为任务成功概率。

实现位置：

| 文件 | 作用 |
|---|---|
| `EasyR1/verl/trainer/beta_thompson_sampler.py` | Beta 后验、目标感知 Thompson Sampling 与 sampler 状态 |
| `EasyR1/verl/trainer/beta_thompson_trainer.py` | rollout 后解析最终答案并更新后验 |
| `EasyR1/verl/trainer/data_loader.py` | 在 Beta 模式下接入 sampler 与稳定数据索引 |
| `EasyR1/verl/trainer/config.py` | Thompson Sampling 配置项 |
| `EasyR1/examples/BETA_THOMPSON_SAMPLING.md` | 详细实验说明 |

在原 Android GUI 训练命令中加入以下参数即可启用：

```bash
data.sampler_type=beta_thompson \
data.beta_ts_target_success=0.5 \
data.beta_ts_temperature=0.15 \
data.beta_ts_prior_alpha=1.0 \
data.beta_ts_prior_beta=1.0 \
data.beta_ts_uniform_mix=0.1 \
algorithm.online_filtering=false
```

训练时会额外记录 `beta_ts/batch_success_rate`、`beta_ts/selected_posterior_mean`、`beta_ts/posterior_mean`、`beta_ts/posterior_min` 和 `beta_ts/posterior_max`。第一组实验建议仅比较 `data.sampler_type=random` 与 `data.sampler_type=beta_thompson`，其余参数保持一致。

当前实现使用过滤后数据集的稳定索引维护后验，同时向 batch 暴露 `task_id`。恢复 checkpoint 时必须保证数据集内容、过滤条件和排序不发生变化。Beta 模式使用单进程 DataLoader，确保本轮后验更新可以影响下一批采样。

## 安装、训练与数据

```bash
cd EasyR1
pip install -e .
bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

Yellow 消歧对比实验：

```bash
YELLOW_DISAMBIGUATION=false bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
YELLOW_DISAMBIGUATION=true EXPERIMENT_NAME=grpo_yellow_disambig bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

训练数据通常为包含截图与 `ground_truth` 位置的 Arrow/Hugging Face 数据集。数据、模型权重和以下大文件不会纳入版本库，需要在本地配置：

- `EasyR1/checkpoints/` 与 `EasyR1/wandb/`
- `Mind2Web/data/`
- 训练脚本中的数据路径、模型路径与 API 环境变量

Mind2Web 当前仍保留为独立数据集与基线目录，尚未接入原生 EasyR1 Dataset/DataLoader。Android GUI 的 Beta Thompson Sampling 是课程采样机制的首个实验实现，后续可以将成功事件扩展为 reward variance、非零 advantage 或 effective-gradient proxy。