"""Task-level Mind2Web dataset adapter for trajectory-level GRPO.

The original Mind2Web action-prediction trainer flattens each task into
independent step-level SFT examples.  GRPO_Curriculum keeps the task intact:
one dataset item is one fixed offline trajectory with states S_1...S_t.  The
rollout adapter later samples multiple action trajectories over those states.
"""

from __future__ import annotations

import json
import pickle
from typing import Any, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data.qwen_vl_compat import expand_text_position_ids_for_qwen_vl
from prompts.mind2web import POLICY_SYSTEM, build_step_state
from verl.utils import torch_functional as VF


class Mind2WebTrajectoryDataset(Dataset):
    """Load Mind2Web as task-level offline trajectories.

    Each returned item still satisfies EasyR1's batch contract by providing a
    small seed prompt (`input_ids`, `raw_prompt_ids`, etc.).  The true rollout
    states are stored in `trajectory_data.steps[*].state_prompt`.
    """

    @staticmethod
    def _resolve_split_files(split_file: str) -> str | list[str]:
        """Allow comma-separated globs, e.g. test_task/*.json,test_website/*.json."""

        if "," not in split_file:
            return split_file
        return [part.strip() for part in split_file.split(",") if part.strip()]

    def __init__(
        self,
        data_path: str,
        split_file: str,
        tokenizer: PreTrainedTokenizer,
        hf_split: str = "train",
        max_prompt_length: int = 2048,
        truncation: str = "right",
        candidate_source: str = "ranked",
        score_file: Optional[str] = None,
        top_k: int = 50,
        max_candidates: int = 20,
        previous_k: int = 5,
        keep_html_brackets: bool = False,
        task_filter: str = "none",
        min_positive_ratio: float = 0.0,
        previous_action_source: str = "gold",
    ) -> None:
        """Mind2Web task-level dataset.

        Args:
            previous_action_source: how the rollout adapter builds the
                "Previous actions" block at step i.
                - "gold": use the dataset's annotated history `action_reprs[:i]`
                  (strict offline; same prompt for every rollout sample).
                - "policy": rebuild from the policy's own sampled actions
                  `context.generated_actions[:i]` (semi-online; trajectories
                  diverge across rollouts; tree_repr/DOM is still fixed).
        """

        if previous_action_source not in {"gold", "policy"}:
            raise ValueError(f"Unsupported previous_action_source: {previous_action_source}")

        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.candidate_source = candidate_source
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.previous_k = previous_k
        self.keep_html_brackets = keep_html_brackets
        self.task_filter = task_filter
        self.min_positive_ratio = min_positive_ratio
        self.previous_action_source = previous_action_source

        self.candidate_results = self._load_candidate_results(score_file)
        resolved_files = self._resolve_split_files(split_file)
        print(f"Loading Mind2Web {hf_split} dataset from: {split_file}", flush=True)
        self.dataset = load_dataset(
            data_path,
            data_files={hf_split: resolved_files},
            split=hf_split,
        )
        print(f"Loaded Mind2Web {hf_split} dataset: {len(self.dataset)} tasks", flush=True)
        self.task_indices = self._build_task_indices()

    @staticmethod
    def _load_candidate_results(score_file: Optional[str]) -> Optional[dict[str, Any]]:
        """Load optional Mind2Web candidate-generation scores/ranks."""

        if not score_file:
            return None
        with open(score_file, "rb") as file:
            return pickle.load(file)

    def _build_task_indices(self) -> list[int]:
        """Apply task-level filters without breaking trajectory structure."""

        task_indices: list[int] = []
        for index, task in enumerate(self.dataset):
            valid_mask = [len(action.get("pos_candidates", [])) > 0 for action in task["actions"]]
            if self.task_filter == "all_positive" and not all(valid_mask):
                continue
            if self.task_filter == "min_positive_ratio":
                ratio = sum(valid_mask) / max(len(valid_mask), 1)
                if ratio < self.min_positive_ratio:
                    continue
            task_indices.append(index)
        return task_indices

    def __len__(self) -> int:
        return len(self.task_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task = self.dataset[self.task_indices[index]]
        trajectory_data = self._build_trajectory_data(task)
        seed_prompt = self._build_seed_prompt(trajectory_data)

        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": POLICY_SYSTEM},
                {"role": "user", "content": seed_prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self.tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

        # Reuse `input_ids` instead of re-encoding `prompt_text` for raw_prompt_ids.
        raw_prompt_ids = input_ids.tolist()

        # Qwen-VL forward requires (4, seq_len) mrope position_ids; for text-only
        # Mind2Web prompts we replicate the text rope across all 4 channels.
        position_ids = expand_text_position_ids_for_qwen_vl(position_ids, self.tokenizer)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = self._truncate_ids(raw_prompt_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            # ground_truth remains a string because the existing reward manager
            # passes it through directly to custom reward functions.
            "ground_truth": json.dumps(trajectory_data["gold_trajectory"], ensure_ascii=False),
            "trajectory_data": trajectory_data,
        }

    def _truncate_ids(self, token_ids: list[int]) -> list[int]:
        """Match EasyR1's truncation behavior for raw_prompt_ids."""

        if self.truncation == "left":
            return token_ids[-self.max_prompt_length :]
        if self.truncation == "right":
            return token_ids[: self.max_prompt_length]
        raise RuntimeError(f"Prompt length {len(token_ids)} is longer than {self.max_prompt_length}.")

    def _build_seed_prompt(self, trajectory_data: dict[str, Any]) -> str:
        """Build a compact task-level seed prompt for EasyR1 bookkeeping.

        The seed prompt is not the per-step policy state.  It gives the trainer
        a conventional prompt tensor while the rollout adapter consumes
        `trajectory_data.steps[*].state_prompt` for actual generation.
        """

        return (
            "You are solving a Mind2Web task with fixed offline webpage states.\n"
            f"Task: {trajectory_data['confirmed_task']}\n"
            f"Website: {trajectory_data['website']}\n"
            f"Number of states: {len(trajectory_data['steps'])}\n"
            "The rollout adapter will ask for the next action at each state."
        )

    def _build_trajectory_data(self, task: dict[str, Any]) -> dict[str, Any]:
        """Convert one raw Mind2Web task into fixed rollout states."""

        steps = []
        gold_trajectory = []
        action_reprs = task.get("action_reprs", [])
        for step_index, action in enumerate(task["actions"]):
            sample = {
                "website": task["website"],
                "confirmed_task": task["confirmed_task"],
                "annotation_id": task["annotation_id"],
                "previous_actions": action_reprs[:step_index],
                "action_uid": action["action_uid"],
                "operation": action["operation"],
                "pos_candidates": self._candidates_with_rank(task["annotation_id"], action, "pos_candidates"),
                "neg_candidates": self._candidates_with_rank(task["annotation_id"], action, "neg_candidates"),
                "cleaned_html": action["cleaned_html"],
            }
            candidate_ids = self._select_candidate_ids(sample)
            state = build_step_state(
                sample,
                candidate_ids=candidate_ids,
                previous_k=self.previous_k,
                keep_html_brackets=self.keep_html_brackets,
            )

            pos_ids = [candidate["backend_node_id"] for candidate in sample["pos_candidates"]]
            target_action = (
                sample["operation"]["op"] + " " + sample["operation"].get("value", "")
            ).strip()
            # Keep only fields downstream code (rollout / reward) needs.  In
            # particular, drop `cleaned_html`, full pos/neg candidate objects:
            # they balloon the DataProto and Ray serialization once carried
            # through `trajectory_data` -> step rows.
            step_payload = {
                "step_index": step_index,
                "action_uid": sample["action_uid"],
                "previous_actions": sample["previous_actions"],
                "candidate_ids": candidate_ids,
                # `tree_repr` is the fixed DOM half of the state prompt; we
                # keep it so the rollout adapter can rebuild `state_prompt`
                # under `previous_action_source="policy"`.
                "tree_repr": state["tree_repr"],
                "state_prompt": state["state_prompt"],
                "choices": state["choices"],
                "pos_ids": pos_ids,
                "operation": sample["operation"],
                "target_action": target_action,
                "seq_target": state["seq_target"],
                "valid_positive": len(pos_ids) > 0,
            }
            steps.append(step_payload)
            gold_trajectory.append(
                {
                    "step_index": step_index,
                    "action_uid": sample["action_uid"],
                    "pos_ids": pos_ids,
                    "operation": sample["operation"],
                    "target_action": target_action,
                    "seq_target": state["seq_target"],
                    "valid_positive": len(pos_ids) > 0,
                }
            )

        return {
            "annotation_id": task["annotation_id"],
            "website": task["website"],
            "domain": task.get("domain"),
            "subdomain": task.get("subdomain"),
            "confirmed_task": task["confirmed_task"],
            "gold_action_reprs": action_reprs,
            "steps": steps,
            "gold_trajectory": gold_trajectory,
            # Rollout adapter reads these to decide whether to keep the
            # dataset's gold state_prompt or rebuild seq_input each step from
            # the policy's own action history.
            "previous_action_source": self.previous_action_source,
            "previous_k": self.previous_k,
        }

    def _candidates_with_rank(self, annotation_id: str, action: dict[str, Any], key: str) -> list[dict[str, Any]]:
        """Attach candidate rank/score from Mind2Web candidate-generation output."""

        candidates = [dict(candidate) for candidate in action.get(key, [])]
        if self.candidate_results is None:
            return candidates

        sample_id = f"{annotation_id}_{action['action_uid']}"
        scores = self.candidate_results.get("scores", {}).get(sample_id, {})
        ranks = self.candidate_results.get("ranks", {}).get(sample_id, {})
        for candidate in candidates:
            candidate_id = candidate["backend_node_id"]
            if candidate_id in scores:
                candidate["score"] = scores[candidate_id]
            if candidate_id in ranks:
                candidate["rank"] = ranks[candidate_id]
        return candidates

    def _select_candidate_ids(self, sample: dict[str, Any]) -> list[str]:
        """Select the candidate ids used to prune the DOM for this fixed state."""

        pos_candidates = sample.get("pos_candidates", [])
        neg_candidates = sample.get("neg_candidates", [])
        if self.candidate_source == "ranked" and self.candidate_results is not None:
            all_candidates = [
                candidate
                for candidate in [*pos_candidates, *neg_candidates]
                if candidate.get("rank", 10**9) < self.top_k
            ]
        elif self.candidate_source in {"raw", "ranked"}:
            # If ranked mode is requested without a score file, fall back to raw
            # candidates so the DOM state is still constructible.
            all_candidates = [*pos_candidates, *neg_candidates]
        else:
            raise ValueError(f"Unsupported Mind2Web candidate_source: {self.candidate_source}")

        candidate_ids: list[str] = []
        for candidate in all_candidates:
            candidate_id = candidate["backend_node_id"]
            if candidate_id not in candidate_ids:
                candidate_ids.append(candidate_id)
            if len(candidate_ids) >= self.max_candidates:
                break
        return candidate_ids
