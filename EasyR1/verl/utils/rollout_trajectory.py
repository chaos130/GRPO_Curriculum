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

"""Per-step rollout trajectory dump (prompt + n rollouts + rewards) for debugging."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from ..protocol import DataProto


def _decode_prompt_response(
    tokenizer: PreTrainedTokenizer,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    skip_special_tokens: bool,
) -> tuple[str, str, int]:
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=skip_special_tokens)
    valid_len = int(response_mask.sum().item())
    valid_response_ids = response_ids[:valid_len]
    response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=skip_special_tokens)
    return prompt_text, response_text, valid_len


def _per_sample_reward(reward_metrics: dict[str, list[Any]], index: int) -> dict[str, Any]:
    sample_reward: dict[str, Any] = {}
    for key, values in reward_metrics.items():
        if index < len(values):
            value = values[index]
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            sample_reward[key] = value
    return sample_reward


def _per_sample_extras(reward_extras: dict[str, list[Any]], index: int) -> dict[str, Any]:
    """从 reward_extras 中按 index 取出非数值 trace；自动剥掉 reward function 约定的 `_` 前缀。"""
    sample_extras: dict[str, Any] = {}
    for key, values in reward_extras.items():
        if index < len(values):
            display_key = key.lstrip("_") or key
            sample_extras[display_key] = values[index]
    return sample_extras


def _per_sample_advantage(
    advantages: Optional[torch.Tensor], response_mask: torch.Tensor, index: int
) -> Optional[float]:
    if advantages is None:
        return None
    mask = response_mask[index].bool()
    if mask.sum() == 0:
        return None
    adv = advantages[index][mask]
    return float(adv.mean().item())


def build_rollout_trajectory_dict(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    global_step: int,
    rollout_n: int,
    rollout_batch_size: int,
    reward_metrics: Optional[dict[str, list[Any]]] = None,
    reward_extras: Optional[dict[str, list[Any]]] = None,
    skip_special_tokens: bool = True,
) -> dict[str, Any]:
    """Group interleaved rollout rows by uid into prompt-level trajectories."""
    reward_metrics = reward_metrics or {}
    reward_extras = reward_extras or {}
    uids = batch.non_tensor_batch["uid"]
    uid_to_indices: dict[str, list[int]] = defaultdict(list)
    uid_order: list[str] = []
    for index, uid in enumerate(uids):
        uid = str(uid)
        if uid not in uid_to_indices:
            uid_order.append(uid)
        uid_to_indices[uid].append(index)

    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    ground_truths = batch.non_tensor_batch.get("ground_truth")
    advantages = batch.batch.get("advantages")

    groups = []
    for group_index, uid in enumerate(uid_order):
        indices = uid_to_indices[uid]
        first = indices[0]
        prompt_text, _, _ = _decode_prompt_response(
            tokenizer, prompts[first], responses[first], response_mask[first], skip_special_tokens
        )
        ground_truth = None
        if ground_truths is not None:
            ground_truth = ground_truths[first]
            if isinstance(ground_truth, np.ndarray):
                ground_truth = ground_truth.item()
            ground_truth = str(ground_truth)

        rollouts = []
        for rollout_index, sample_index in enumerate(indices):
            _, response_text, response_token_length = _decode_prompt_response(
                tokenizer,
                prompts[sample_index],
                responses[sample_index],
                response_mask[sample_index],
                skip_special_tokens,
            )
            overall_score = None
            if "token_level_scores" in batch.batch:
                valid_len = int(response_mask[sample_index].sum().item())
                if valid_len > 0:
                    overall_score = float(
                        batch.batch["token_level_scores"][sample_index, valid_len - 1].item()
                    )

            rollout_item = {
                "rollout_index": rollout_index,
                "sample_index": int(sample_index),
                "response": response_text,
                "response_token_length": response_token_length,
                "overall_score": overall_score,
                "reward": _per_sample_reward(reward_metrics, sample_index),
                "advantage_mean": _per_sample_advantage(advantages, response_mask, sample_index),
            }
            extras = _per_sample_extras(reward_extras, sample_index)
            if extras:
                rollout_item["extras"] = extras
            rollouts.append(rollout_item)

        groups.append(
            {
                "group_index": group_index,
                "group_id": uid,
                "prompt": prompt_text,
                "ground_truth": ground_truth,
                "rollouts": rollouts,
            }
        )

    return {
        "global_step": global_step,
        "rollout_n": rollout_n,
        "rollout_batch_size": rollout_batch_size,
        "num_prompts": len(groups),
        "num_rollout_samples": len(uids),
        "groups": groups,
    }


def save_rollout_trajectory_json(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    global_step: int,
    rollout_n: int,
    rollout_batch_size: int,
    output_dir: str,
    reward_metrics: Optional[dict[str, list[Any]]] = None,
    reward_extras: Optional[dict[str, list[Any]]] = None,
    skip_special_tokens: bool = True,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    payload = build_rollout_trajectory_dict(
        batch=batch,
        tokenizer=tokenizer,
        global_step=global_step,
        rollout_n=rollout_n,
        rollout_batch_size=rollout_batch_size,
        reward_metrics=reward_metrics,
        reward_extras=reward_extras,
        skip_special_tokens=skip_special_tokens,
    )
    output_path = os.path.join(output_dir, f"step_{global_step:04d}.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"[rollout_trajectory] Saved JSON to {output_path}")
    return output_path
