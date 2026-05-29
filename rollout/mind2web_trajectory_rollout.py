"""Offline trajectory rollout for Mind2Web over fixed dataset states.

EasyR1's native rollout is one prompt -> one response.  Mind2Web trajectory
GRPO needs one task -> multiple full action trajectories, where the webpage
states S_1...S_t are fixed by the offline dataset.  This adapter repeatedly
calls the existing verl/vLLM generation backend, one state step at a time, then
returns a normal DataProto so actor/reward/advantage code can stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from data.qwen_vl_compat import expand_text_position_ids_for_qwen_vl
from prompts.mind2web import POLICY_SYSTEM, build_seq_input, build_step_prompt
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.utils import torch_functional as VF


def _object_array(items: list[Any]) -> np.ndarray:
    """Create a 1-D object array without NumPy expanding nested lists/dicts."""

    array = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        array[index] = item
    return array


@dataclass
class _TrajectoryContext:
    """Mutable state for one sampled trajectory during rollout."""

    task_index: int
    rollout_index: int
    uid: str
    trajectory_id: str
    trajectory_data: dict[str, Any]
    generated_actions: list[str] = field(default_factory=list)


def _state_prompt_for_step(
    context: "_TrajectoryContext",
    step_index: int,
    step: dict[str, Any],
) -> str:
    """Build the policy prompt for `step_index` under the configured source.

    - "gold" (default): reuse the dataset's prebuilt `state_prompt`; identical
      across all rollouts of the same task, matching Mind2Web SFT semantics.
    - "policy": keep the fixed DOM (`tree_repr`) but rebuild `seq_input` from
      this rollout's own previously sampled actions. Trajectories of the same
      task then diverge as soon as their `generated_actions` differ.
    """

    source = context.trajectory_data.get("previous_action_source", "gold")
    if source == "gold" or step_index == 0:
        return step["state_prompt"]

    previous_k = int(context.trajectory_data.get("previous_k", 5))
    return build_step_prompt(
        tree_repr=step["tree_repr"],
        seq_input=build_seq_input(
            confirmed_task=context.trajectory_data["confirmed_task"],
            previous_actions=context.generated_actions[:step_index],
            previous_k=previous_k,
        ),
    )


def _encode_prompt_batch(
    prompts: list[str],
    tokenizer,
    max_prompt_length: int,
    truncation: str,
    meta_info: dict[str, Any],
) -> DataProto:
    """Encode per-step Mind2Web prompts into EasyR1 generation tensors."""

    input_ids_list, attention_mask_list, position_ids_list, raw_prompt_ids = [], [], [], []
    for prompt in prompts:
        chat_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": POLICY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = tokenizer([chat_prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

        # Reuse `input_ids` instead of re-encoding the same `chat_prompt`.
        prompt_ids = input_ids.tolist()

        # Qwen-VL forward needs (4, seq_len) mrope position_ids even for text-only inputs.
        position_ids = expand_text_position_ids_for_qwen_vl(position_ids, tokenizer)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        if len(prompt_ids) > max_prompt_length:
            if truncation == "left":
                prompt_ids = prompt_ids[-max_prompt_length:]
            elif truncation == "right":
                prompt_ids = prompt_ids[:max_prompt_length]
            else:
                raise RuntimeError(f"Prompt length {len(prompt_ids)} exceeds {max_prompt_length}.")

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
        raw_prompt_ids.append(prompt_ids)

    return DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "position_ids": torch.stack(position_ids_list, dim=0),
        },
        non_tensors={"raw_prompt_ids": _object_array(raw_prompt_ids)},
        meta_info=meta_info,
    )


def _decode_generation_rows(step_output: DataProto, tokenizer, skip_special_tokens: bool) -> list[str]:
    """Decode one generated action per active trajectory row."""

    response_ids = step_output.batch["responses"]
    response_lengths = torch.sum(step_output.batch["response_mask"], dim=-1)
    decoded: list[str] = []
    for index in range(len(step_output)):
        valid_len = int(response_lengths[index].item())
        decoded.append(
            tokenizer.decode(
                response_ids[index][:valid_len],
                skip_special_tokens=skip_special_tokens,
            )
        )
    return decoded


def _attach_step_metadata(
    step_output: DataProto,
    active_rows: list[tuple[_TrajectoryContext, dict[str, Any]]],
    task_ground_truths: np.ndarray,
) -> None:
    """Attach task/trajectory metadata to generated step rows.

    Per-state GRPO: ``uid`` groups rows that share the same fixed state (one task step).
    ``rollout.n`` rows per group compare at that state.  ``task_uid`` keeps the batch-level
    task id for rollout_batch_size counting and trajectory JSON dumps.
    """

    step_output.non_tensor_batch["task_uid"] = _object_array([row[0].uid for row in active_rows])
    step_output.non_tensor_batch["uid"] = _object_array(
        [f"{row[0].uid}:{row[1]['step_index']}" for row in active_rows]
    )
    step_output.non_tensor_batch["trajectory_id"] = _object_array([row[0].trajectory_id for row in active_rows])
    step_output.non_tensor_batch["rollout_index"] = _object_array([row[0].rollout_index for row in active_rows])
    step_output.non_tensor_batch["step_index"] = _object_array([row[1]["step_index"] for row in active_rows])
    step_output.non_tensor_batch["action_uid"] = _object_array([row[1]["action_uid"] for row in active_rows])
    step_output.non_tensor_batch["step_data"] = _object_array([row[1] for row in active_rows])
    step_output.non_tensor_batch["trajectory_data"] = _object_array([row[0].trajectory_data for row in active_rows])
    step_output.non_tensor_batch["ground_truth"] = _object_array(
        [task_ground_truths[row[0].task_index] for row in active_rows]
    )


def _attach_complete_trajectory_metadata(
    step_outputs: list[tuple[DataProto, list[tuple[_TrajectoryContext, dict[str, Any]]]]]
) -> None:
    """Store the complete sampled action trajectory on every generated step row."""

    for step_output, active_rows in step_outputs:
        step_output.non_tensor_batch["predicted_trajectory"] = _object_array(
            [list(row[0].generated_actions) for row in active_rows]
        )


def generate_mind2web_trajectory_batch(
    *,
    actor_rollout_ref_wg,
    task_batch: DataProto,
    tokenizer,
    rollout_n: int,
    max_prompt_length: int,
    truncation: str,
    generation_meta: dict[str, Any],
    skip_special_tokens: bool = True,
) -> DataProto:
    """Generate `rollout_n` fixed-state trajectories for each Mind2Web task.

    The returned DataProto is expanded to step-action rows:
    `batch_size = num_tasks * rollout_n * num_steps_per_task` (summed over
    variable-length tasks).  Each row is a standard EasyR1 prompt/response pair,
    while metadata identifies which task trajectory it belongs to.
    """

    trajectory_data = task_batch.non_tensor_batch["trajectory_data"]
    task_ground_truths = task_batch.non_tensor_batch["ground_truth"]
    task_uids = task_batch.non_tensor_batch["uid"]

    contexts: list[_TrajectoryContext] = []
    for task_index, task in enumerate(trajectory_data):
        for rollout_index in range(rollout_n):
            uid = str(task_uids[task_index])
            contexts.append(
                _TrajectoryContext(
                    task_index=task_index,
                    rollout_index=rollout_index,
                    uid=uid,
                    trajectory_id=f"{uid}:{rollout_index}",
                    trajectory_data=task,
                )
            )

    max_steps = max(len(context.trajectory_data["steps"]) for context in contexts)
    step_outputs: list[tuple[DataProto, list[tuple[_TrajectoryContext, dict[str, Any]]]]] = []
    base_meta = dict(generation_meta)
    # We manually create rollout_n trajectories, so each per-step vLLM call must
    # sample exactly one action per active trajectory row.
    base_meta["n"] = 1
    # Same fixed state_prompt is fed by every rollout_index, so a deterministic
    # vLLM seed would collapse all `rollout_n` samples to the identical text and
    # zero the GRPO advantage.  Force per-request random seeds.
    base_meta["seed"] = None

    for step_index in range(max_steps):
        active_rows: list[tuple[_TrajectoryContext, dict[str, Any]]] = []
        prompts: list[str] = []
        for context in contexts:
            steps = context.trajectory_data["steps"]
            if step_index >= len(steps):
                continue
            step = steps[step_index]
            active_rows.append((context, step))
            prompts.append(_state_prompt_for_step(context, step_index, step))

        if not active_rows:
            continue

        step_batch = _encode_prompt_batch(
            prompts=prompts,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            truncation=truncation,
            meta_info=base_meta,
        )
        step_batch, pad_size = pad_dataproto_to_divisor(step_batch, actor_rollout_ref_wg.world_size)
        step_output = actor_rollout_ref_wg.generate_sequences(step_batch)
        step_output = unpad_dataproto(step_output, pad_size=pad_size)
        decoded_actions = _decode_generation_rows(step_output, tokenizer, skip_special_tokens)
        for (context, _), action in zip(active_rows, decoded_actions):
            context.generated_actions.append(action)

        _attach_step_metadata(step_output, active_rows, task_ground_truths)
        step_outputs.append((step_output, active_rows))

    if not step_outputs:
        raise RuntimeError("Mind2Web trajectory rollout produced no step outputs.")

    _attach_complete_trajectory_metadata(step_outputs)
    return DataProto.concat([output for output, _ in step_outputs])

