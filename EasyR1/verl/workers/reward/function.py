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

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig

# 约定：reward function 返回的 score dict 中，以 "_" 开头的 key 视为非数值 trace
# （如 LLM-as-Judge 的 prompt / raw response），不会进入 reward_metrics（避免 np.mean 报错），
# 而是被拆到 reward_extras 里，由 trajectory dumper 写进 step JSON。
_EXTRA_KEY_PREFIX = "_"


class RewardInput(TypedDict, total=False):
    response: str
    response_length: int
    ground_truth: str
    uid: Optional[str]   # 同一个 prompt 的 N 个 rollout 共享同一 uid；reward function 可据此分组
    images: Optional[list]  # PIL.Image 列表（或路径），从 data.non_tensor_batch["multi_modal_data"] 提取


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


def _push_score(
    score: dict,
    reward_metrics: dict[str, list[Any]],
    reward_extras: dict[str, list[Any]],
) -> None:
    """把单条样本的 score dict 拆到 metrics（数值，进 wandb）和 extras（trace，进 JSON dump）。"""
    for key, value in score.items():
        if key.startswith(_EXTRA_KEY_PREFIX):
            reward_extras[key].append(value)
        else:
            reward_metrics[key].append(value)


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(
        self, data: DataProto
    ) -> Tuple[torch.Tensor, dict[str, list[Any]], dict[str, list[Any]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics: dict[str, list[Any]] = defaultdict(list)
        reward_extras: dict[str, list[Any]] = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        uids = data.non_tensor_batch.get("uid")
        mmd = data.non_tensor_batch.get("multi_modal_data")
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            images_i = None
            if mmd is not None and isinstance(mmd[i], dict):
                images_i = mmd[i].get("images")
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "uid": str(uids[i]) if uids is not None else f"sample_{i}",
                    "images": images_i,
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            _push_score(score, reward_metrics, reward_extras)

        return reward_tensor, reward_metrics, reward_extras


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(
        self, data: DataProto
    ) -> Tuple[torch.Tensor, dict[str, list[Any]], dict[str, list[Any]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        uids = data.non_tensor_batch.get("uid")
        mmd = data.non_tensor_batch.get("multi_modal_data")
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            images_i = None
            if mmd is not None and isinstance(mmd[i], dict):
                images_i = mmd[i].get("images")
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "uid": str(uids[i]) if uids is not None else f"sample_{i}",
                    "images": images_i,
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics: dict[str, list[Any]] = defaultdict(list)
        reward_extras: dict[str, list[Any]] = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            _push_score(score, reward_metrics, reward_extras)

        return reward_tensor, reward_metrics, reward_extras


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(
        self, data: DataProto
    ) -> Tuple[torch.Tensor, dict[str, list[Any]], dict[str, list[Any]]]:
        """Compute reward for a batch of data.

        Returns:
            reward_tensor   : 每个 token 的 reward（GAE 用）
            reward_metrics  : 进 wandb 的数值 metrics
            reward_extras   : 非数值 trace（如 LLM-as-Judge 原始 prompt / response），由 reward function 通过 `_xxx` 前缀 key 提交
        """
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
