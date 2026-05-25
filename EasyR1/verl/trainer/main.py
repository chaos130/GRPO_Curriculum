# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        # Ray 子进程的环境变量（shell 里 export 的变量不会自动传给 Ray worker，必须在这里显式声明）
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            # 显存碎片优化：必须为 False，因为 vLLM 的 CuMemAllocator 不兼容 expandable_segments
            # 参考：https://github.com/pytorch/pytorch/issues/147851
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            # 禁用 GPU P2P 直连：消费级显卡（RTX 4090/3090）无 NVLink，
            # 不禁用会报错：Cuda failure 217 'peer access is not supported between these two devices'
            "NCCL_P2P_DISABLE": "1",
            # 禁用 InfiniBand 探测：单机多卡训练用不到 IB，避免误检测警告
            "NCCL_IB_DISABLE": "1",
            # 显式启用共享内存通信（默认值，写出来更稳定）
            "NCCL_SHM_DISABLE": "0",
            # 禁用 CUDA Multicast Memory：NCCL 2.19+ 版本即使 P2P_DISABLE=1，
            # CUMEM 仍会尝试 cudaDeviceEnablePeerAccess，在消费级 4090 上报 217 错误
            "NCCL_CUMEM_ENABLE": "0",
            # NCCL 2.24+ 默认开启 host cuMem，无 NVLink/Docker 多卡时在 barrier 处 SIGSEGV
            "NCCL_CUMEM_HOST_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
        }
        # 转发 LLM-as-Judge / wandb 相关 env 到 Ray worker（reward function 跑在 Ray actor 里）
        for key in (
            "JUDGE_ENABLED",
            "JUDGE_API_KEY",
            "JUDGE_BASE_URL",
            "JUDGE_MODEL",
            "JUDGE_TIMEOUT_S",
            "JUDGE_MAX_WORKERS",
            "JUDGE_TEMPERATURE",
            "OPENAI_API_KEY",
            "WANDB_API_KEY",
            "WANDB_DIR",
            "WANDB_PROJECT",
        ):
            value = os.environ.get(key)
            if value is not None:
                env_vars[key] = value
        ray.init(runtime_env={"env_vars": env_vars})

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
