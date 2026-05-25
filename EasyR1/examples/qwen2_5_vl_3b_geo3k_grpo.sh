#!/bin/bash
# ========== 环境变量配置 ==========

# [GPU通信] 禁用 NCCL P2P 通信
# 适用：消费级 GPU（RTX 4090/3090 等）之间没有 NVLink，无法直接 P2P 传输
# 不禁用会报错：Cuda failure 217 'peer access is not supported between these two devices'
# 注意：这个变量必须同时在 verl/trainer/main.py 的 runtime_env 里设置，否则不会传给 Ray 子进程
export NCCL_P2P_DISABLE=1

# [GPU通信] 禁用 InfiniBand 高速网络
# 适用：单机训练用不到 IB，避免 NCCL 误检测导致警告或卡住
export NCCL_IB_DISABLE=1

# [显存优化] 关闭可扩展显存段
# 注意：vLLM 的 CuMemAllocator 与 expandable_segments:True 不兼容，会触发 AssertionError
# 参考：https://github.com/pytorch/pytorch/issues/147851
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# [HuggingFace 加速] 禁用 hf_transfer 加速下载
# 原因：hf_transfer 走 cas-bridge 节点，国内网络无法访问
export HF_HUB_ENABLE_HF_TRANSFER=0

# [HuggingFace 镜像] 走国内镜像下载（数据集、tokenizer 等会用到）
export HF_ENDPOINT=https://hf-mirror.com

# [HuggingFace 离线模式] 强制 transformers 库只读取本地文件，不走网络
# 必要：本地模型路径需要 local_files_only=True 才能跳过 HF Hub 仓库 ID 校验
export TRANSFORMERS_OFFLINE=1

# [Weights & Biases] 实验追踪 API Key
# 在 https://wandb.ai/authorize 获取
export WANDB_API_KEY=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

# ========== 训练参数配置 ==========

# 本地模型路径（已通过 docker -v 挂载进容器）
MODEL_PATH=/workspace/model/Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/workspace/data/geometry3k/data/train-00000-of-00001.parquet \
    data.val_files=/workspace/data/geometry3k/data/test-00000-of-00001.parquet \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=32 \
    worker.actor.model.lora.exclude_modules='.*visual.*' \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.gpu_memory_utilization=0.25 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.n=3 \
    worker.rollout.enforce_eager=true \
    worker.actor.global_batch_size=64 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.ref.offload.offload_params=true \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo_0508_01 \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=false

# ========== 训练参数说明 ==========
# data.max_prompt_length=1024 / max_response_length=1024
#   将序列长度从默认 2048 砍半，反向传播激活值显存直接减半（最有效的省显存手段）
# data.rollout_batch_size=256
#   一次 rollout 采样 256 条 prompt（默认 512）
# worker.actor.model.lora.rank=32
#   开启 LoRA 训练（不再全参数训练），优化器状态显存占用从“按全模型”降到“按 LoRA 参数”
# worker.actor.optim.strategy=adamw_bf16
#   使用 bf16 优化器状态，进一步降低 Adam 的 moment 显存
# worker.rollout.gpu_memory_utilization=0.25
#   vLLM 推理时占用 GPU 显存的比例（24GB × 0.25 = 6GB）
# worker.rollout.tensor_parallel_size=2
#   vLLM 推理模型切分到 2 张卡（每卡只放一半权重）
# worker.rollout.n=3
#   每个 prompt 生成 3 条 response 用于 GRPO 组内比较（默认 5）
#   GRPO 最少需要 2 条，3 条已能保证训练效果
# worker.rollout.enforce_eager=true
#   关闭 vLLM CUDA Graph，节省 1-2GB 显存（推理慢一些但不影响训练）
# worker.actor.global_batch_size=64
#   一次完整梯度更新的样本数（默认 128），数据量减半相应降低
# worker.actor.micro_batch_size_per_device_for_update=1
#   训练时每卡每次只处理 1 条样本做梯度更新（峰值显存最低）
# worker.actor.micro_batch_size_per_device_for_experience=1
#   生成 log_prob 时每卡 1 条样本
# worker.actor.offload.offload_params=true        Actor 参数卸载到 CPU
# worker.actor.offload.offload_optimizer=true     Adam 优化器状态卸载到 CPU
# worker.ref.offload.offload_params=true          Ref 模型参数卸载到 CPU
# trainer.n_gpus_per_node=2        本机使用 2 张 GPU
# trainer.val_before_train=false   跳过训练前的初始验证（节省启动时间和显存峰值）
