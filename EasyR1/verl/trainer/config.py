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
"""
PPO config
"""

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    format_prompt_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Extra variables passed to the format_prompt Jinja template (e.g. yellow_disambiguation=true)."""
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    sampler_type: str = "random"
    """Training sampler: `random`, `sequential`, or target-aware `beta_thompson`."""
    task_id_key: str = "task_id"
    """Optional stable dataset column used for logging task identity."""
    beta_ts_target_success: float = 0.5
    """Success probability preferred by target-aware Thompson sampling."""
    beta_ts_temperature: float = 0.15
    """Smaller values concentrate sampling more strongly around the target."""
    beta_ts_prior_alpha: float = 1.0
    beta_ts_prior_beta: float = 1.0
    beta_ts_uniform_mix: float = 0.1
    """Uniform exploration mixed into Thompson probabilities."""

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes"""
    n_gpus_per_node: int = 8
    """number of gpus per node"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation during training"""
    val_before_train: bool = True
    val_after_train: bool = True
    val_only: bool = False
    val_generations_to_log: int = 0
    log_rollout_trajectory_json: bool = False
    rollout_trajectory_json_steps: Optional[List[int]] = None
    save_freq: int = -1
    save_limit: int = -1
    save_model_only: bool = False
    save_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None
    ray_timeline: Optional[str] = None
    find_last_checkpoint: bool = True

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")
        if isinstance(self.rollout_trajectory_json_steps, str):
            self.rollout_trajectory_json_steps = json.loads(self.rollout_trajectory_json_steps)


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
