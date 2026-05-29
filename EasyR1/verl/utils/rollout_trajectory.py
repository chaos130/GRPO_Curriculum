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
    """Group interleaved rollout rows by uid into prompt-level trajectories.

    If `trajectory_id` and `step_index` are present in `non_tensor_batch`
    (e.g. Mind2Web trajectory rollout), use three-level grouping:
    ``task (uid) -> trajectory_id -> step_index``.
    """
    reward_metrics = reward_metrics or {}
    reward_extras = reward_extras or {}
    if "trajectory_id" in batch.non_tensor_batch and "step_index" in batch.non_tensor_batch:
        return _build_trajectory_grouped_dict(
            batch=batch,
            tokenizer=tokenizer,
            global_step=global_step,
            rollout_n=rollout_n,
            rollout_batch_size=rollout_batch_size,
            reward_metrics=reward_metrics,
            reward_extras=reward_extras,
            skip_special_tokens=skip_special_tokens,
        )
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


def _extract_task_instruction(prompt_text: str) -> Optional[str]:
    marker = "Task:"
    if marker not in prompt_text:
        return None
    start = prompt_text.index(marker) + len(marker)
    end = prompt_text.find("\nPrevious actions:", start)
    if end == -1:
        end = prompt_text.find("\nWhat should be the next action?", start)
    if end == -1:
        return prompt_text[start:].strip()
    return prompt_text[start:end].strip()


def _extract_previous_actions(prompt_text: str) -> list[str]:
    marker = "Previous actions:\n"
    if marker not in prompt_text:
        return []
    start = prompt_text.index(marker) + len(marker)
    end = prompt_text.find("What should be the next action?", start)
    block = prompt_text[start:end] if end != -1 else prompt_text[start:]
    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
    if len(lines) == 1 and lines[0].lower() == "none":
        return []
    return lines


def _build_step_timeline(steps_out: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline = []
    for step in steps_out:
        prompt_text = step.get("state_prompt", "")
        gold = step.get("gold") or {}
        reward = step.get("reward") or {}
        timeline.append(
            {
                "step_index": step["step_index"],
                "previous_actions": _extract_previous_actions(prompt_text),
                "predicted_action": step.get("response"),
                "gold_action": gold.get("target_action"),
                "gold_seq_target": gold.get("seq_target"),
                "reward_overall": reward.get("overall", step.get("overall_score")),
                "advantage_mean": step.get("advantage_mean"),
            }
        )
    return timeline


def _build_trajectory_grouped_dict(
    *,
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    global_step: int,
    rollout_n: int,
    rollout_batch_size: int,
    reward_metrics: dict[str, list[Any]],
    reward_extras: dict[str, list[Any]],
    skip_special_tokens: bool,
) -> dict[str, Any]:
    """Mind2Web-style dump: task -> trajectory -> step.

    Expects `non_tensor_batch` to carry `uid`, `trajectory_id`, `step_index`.
    When present, `task_uid` groups rows by Mind2Web task (`uid` is per-state GRPO key).
    Optional fields used when present: `rollout_index`, `step_data`,
    `predicted_trajectory`, `ground_truth`.
    """

    task_uids = batch.non_tensor_batch.get("task_uid", batch.non_tensor_batch["uid"])
    trajectory_ids = batch.non_tensor_batch["trajectory_id"]
    step_indices = batch.non_tensor_batch["step_index"]
    rollout_indices = batch.non_tensor_batch.get("rollout_index")
    step_datas = batch.non_tensor_batch.get("step_data")
    predicted_trajectories = batch.non_tensor_batch.get("predicted_trajectory")
    ground_truths = batch.non_tensor_batch.get("ground_truth")

    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    advantages = batch.batch.get("advantages")

    # task -> trajectory -> [step rows], preserving first-seen order.
    task_order: list[str] = []
    task_to_traj_order: dict[str, list[str]] = defaultdict(list)
    nested: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for sample_index, (task_uid, traj_id) in enumerate(zip(task_uids, trajectory_ids)):
        uid = str(task_uid)
        traj_id = str(traj_id)
        if uid not in nested:
            task_order.append(uid)
        if traj_id not in nested[uid]:
            task_to_traj_order[uid].append(traj_id)
        nested[uid][traj_id].append(sample_index)

    tasks_out = []
    for task_idx, uid in enumerate(task_order):
        # Task-level ground_truth (gold trajectory JSON) — same for all rows of this task.
        ground_truth = None
        if ground_truths is not None:
            sample_index = nested[uid][task_to_traj_order[uid][0]][0]
            ground_truth = ground_truths[sample_index]
            if isinstance(ground_truth, np.ndarray):
                ground_truth = ground_truth.item()
            ground_truth = str(ground_truth)

        trajectories_out = []
        for traj_id in task_to_traj_order[uid]:
            sample_indices = nested[uid][traj_id]
            # Sort by step_index so trajectory reads s0,a0,s1,a1,...
            sample_indices.sort(key=lambda i: int(step_indices[i]))

            steps_out = []
            for sample_index in sample_indices:
                prompt_text, response_text, response_token_length = _decode_prompt_response(
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

                step_info: dict[str, Any] = {
                    "step_index": int(step_indices[sample_index]),
                    "sample_index": int(sample_index),
                    "state_prompt": prompt_text,
                    "response": response_text,
                    "response_token_length": response_token_length,
                    "overall_score": overall_score,
                    "reward": _per_sample_reward(reward_metrics, sample_index),
                    "advantage_mean": _per_sample_advantage(advantages, response_mask, sample_index),
                }
                if step_datas is not None and isinstance(step_datas[sample_index], dict):
                    sd = step_datas[sample_index]
                    # Carry only the small per-step gold pointers; the raw DOM
                    # was already stripped at dataset build time.
                    step_info["gold"] = {
                        "target_action": sd.get("target_action"),
                        "seq_target": sd.get("seq_target"),
                        "pos_ids": sd.get("pos_ids"),
                        "valid_positive": sd.get("valid_positive"),
                    }
                extras = _per_sample_extras(reward_extras, sample_index)
                if extras:
                    step_info["extras"] = extras
                steps_out.append(step_info)

            # rollout_index for the trajectory header (any step's value is consistent).
            rollout_index = None
            if rollout_indices is not None and sample_indices:
                rollout_index = int(rollout_indices[sample_indices[0]])

            predicted_trajectory = None
            if predicted_trajectories is not None and sample_indices:
                pt = predicted_trajectories[sample_indices[0]]
                if isinstance(pt, (list, tuple)):
                    predicted_trajectory = list(pt)

            trajectories_out.append(
                {
                    "trajectory_id": traj_id,
                    "rollout_index": rollout_index,
                    "predicted_trajectory": predicted_trajectory,
                    "timeline": _build_step_timeline(steps_out),
                    "steps": steps_out,
                }
            )

        task_instruction = None
        if trajectories_out and trajectories_out[0]["steps"]:
            task_instruction = _extract_task_instruction(trajectories_out[0]["steps"][0]["state_prompt"])

        tasks_out.append(
            {
                "task_index": task_idx,
                "task_uid": uid,
                "task_instruction": task_instruction,
                "ground_truth": ground_truth,
                "trajectories": trajectories_out,
            }
        )

    return {
        "global_step": global_step,
        "rollout_n": rollout_n,
        "rollout_batch_size": rollout_batch_size,
        "num_tasks": len(tasks_out),
        "num_rollout_samples": len(uids),
        "tasks": tasks_out,
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
