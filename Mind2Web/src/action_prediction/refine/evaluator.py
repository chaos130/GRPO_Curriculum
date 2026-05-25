"""Refinement-based action evaluator for Mind2Web.

One `trajectory_step` == one sample in the flattened dataset.

For each step:
  - Fixed inputs: task, cropped_html (pruned DOM + candidates), trajectory_history
  - refinement_history = []
  - For round = 1..max_rounds:
        policy(task, step, html, history, ref_history, previous_feedback) -> raw
        judge (task, step, round, html, history, ref_history, raw) -> scores, feedback
        append to ref_history
        if total_score(scores) >= threshold: break
  - final_prediction = ref_history[-1]            (user selected "last")
  - Evaluate element_acc / action_f1 / step_acc against ground truth.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from dataloader import format_input_generation

from .judge import JudgeClient, parse_judge_output  # noqa: F401
from .prompts import build_judge_messages, build_policy_messages

logger = logging.getLogger(__name__)


# -------------------------- output parsing helpers -------------------------- #

_ELEMENT_RE = re.compile(r"Element:\s*(.*)", re.IGNORECASE)
_ID_RE = re.compile(r"id\s*=\s*(\d+)")
_ACTION_RE = re.compile(r"Action:\s*(CLICK|SELECT|TYPE|NONE)", re.IGNORECASE)
_VALUE_RE = re.compile(r"Value:\s*(.*)$", re.MULTILINE)


def parse_policy_output(
    text: str, choices: List[List[str]]
) -> Tuple[Optional[str], str, str, str]:
    """Parse a policy LLM response in the generation format.

    Returns (backend_node_id or None, action_op, value, action_str)
      - action_str is the concatenation "<OP> <VALUE>" (stripped) used for F1.
    If the element is 'None' or cannot be matched, backend_node_id is None.
    """
    text = (text or "").strip()

    action_match = _ACTION_RE.search(text)
    action_op = action_match.group(1).upper() if action_match else ""
    if action_op == "NONE":
        action_op = ""

    value_match = _VALUE_RE.search(text)
    value = value_match.group(1).strip() if value_match else ""

    elem_match = _ELEMENT_RE.search(text)
    elem_line = elem_match.group(1).strip() if elem_match else ""

    if not elem_line or elem_line.lower().startswith("none"):
        return None, action_op, value, (action_op + " " + value).strip()

    selected_idx: Optional[int] = None
    id_match = _ID_RE.search(elem_line)
    if id_match is not None:
        idx = int(id_match.group(1))
        if 0 <= idx < len(choices):
            selected_idx = idx

    if selected_idx is None and choices:
        snippets = [c[1] if len(c) > 1 else "" for c in choices]
        scores = [SequenceMatcher(None, elem_line, s).ratio() for s in snippets]
        selected_idx = int(np.argmax(scores)) if scores else None

    backend_node_id = choices[selected_idx][0] if selected_idx is not None else None
    action_str = (action_op + " " + value).strip()
    return backend_node_id, action_op, value, action_str


def _tokenize_for_f1(s: str) -> set:
    toks = s.strip().split()
    return {t for t in toks if t not in string.punctuation}


def calculate_f1(pred: str, label: str) -> float:
    p = _tokenize_for_f1(pred)
    l = _tokenize_for_f1(label)
    if not p and not l:
        return 1.0
    if not p or not l:
        return 0.0
    tp = len(p & l)
    fp = len(p - l)
    fn = len(l - p)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


# ------------------------------- evaluator --------------------------------- #


class RefinementActionEvaluator:
    def __init__(
        self,
        policy_engine,
        judge_engine,
        *,
        max_rounds: int = 4,
        score_threshold: float = 4.0,
        top_k: int = 50,
        max_candidates: int = 20,
        previous_k: int = 5,
        policy_max_new_tokens: int = 128,
        judge_max_new_tokens: int = 256,
        policy_temperature: float = 0.0,
        policy_temperature_boost: float = 0.1,
        judge_temperature: float = 0.0,
        seed: int = 123,
        keep_html_brackets: bool = True,
        num_workers: int = 1,
    ) -> None:
        self.policy = policy_engine
        self.judge = JudgeClient(
            judge_engine,
            max_new_tokens=judge_max_new_tokens,
            temperature=judge_temperature,
        )
        self.max_rounds = max_rounds
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.previous_k = previous_k
        self.policy_max_new_tokens = policy_max_new_tokens
        self.policy_temperature = policy_temperature
        self.policy_temperature_boost = policy_temperature_boost
        self.keep_html_brackets = keep_html_brackets
        self.seed = seed
        self.num_workers = max(1, int(num_workers))

    # ----------------------- helpers ----------------------- #

    def _build_step_context(self, sample: Dict[str, Any]):
        """Build the per-step immutable context: tree_repr, choices, pos_ids,
        target action string."""
        pos_candidates = [c for c in sample["pos_candidates"] if c.get("rank", 0) < self.top_k]
        pos_ids = [c["backend_node_id"] for c in pos_candidates]

        neg_candidates = [c for c in sample["neg_candidates"] if c.get("rank", 0) < self.top_k]
        neg_ids = [c["backend_node_id"] for c in neg_candidates]

        candidate_ids = pos_ids + neg_ids
        if len(candidate_ids) > self.max_candidates:
            # keep all positives, truncate negatives
            keep_negs = self.max_candidates - len(pos_ids)
            keep_negs = max(0, keep_negs)
            candidate_ids = pos_ids + neg_ids[:keep_negs]
        local_rng = random.Random(
            f"{self.seed}_{sample.get('annotation_id', '')}_{sample.get('action_uid', '')}"
        )
        local_rng.shuffle(candidate_ids)

        tree_repr, seq_input, seq_target, choices = format_input_generation(
            sample,
            candidate_ids,
            gt=pos_ids[0] if pos_ids else -1,
            previous_k=self.previous_k,
            keep_html_brackets=self.keep_html_brackets,
        )

        op = sample["operation"]["op"]
        value = sample["operation"].get("value", "")
        target_action_str = (op + " " + value).strip() if op else ""

        return {
            "tree_repr": tree_repr,
            "seq_input": seq_input,
            "choices": choices,
            "pos_ids": pos_ids,
            "target_action_str": target_action_str,
            "target_op": op,
            "target_value": value,
        }

    # ----------------------- main loop per step ----------------------- #

    def _refine_one_step(
        self,
        sample: Dict[str, Any],
        ctx: Dict[str, Any],
        trajectory_step_idx: int,
    ) -> Dict[str, Any]:
        task = sample["confirmed_task"]
        previous_actions = sample.get("previous_actions", [])

        refinement_history: List[Dict[str, Any]] = []
        previous_feedback: Optional[str] = None
        final_entry: Optional[Dict[str, Any]] = None

        for rnd in range(1, self.max_rounds + 1):
            # ---- adaptive temperature: deterministic round 1, explore later ---- #
            if rnd == 1:
                current_temp = self.policy_temperature
            else:
                current_temp = min(
                    1.0,
                    self.policy_temperature + self.policy_temperature_boost * (rnd - 1),
                )

            # ---- policy ---- #
            policy_msgs = build_policy_messages(
                task=task,
                tree_repr=ctx["tree_repr"],
                previous_actions=previous_actions,
                candidate_choices=ctx["choices"],
                refinement_history=refinement_history,
                previous_feedback=previous_feedback,
                trajectory_step_idx=trajectory_step_idx,
                refinement_round=rnd,
            )
            try:
                policy_out = self.policy.generate(
                    prompt=policy_msgs,
                    max_new_tokens=self.policy_max_new_tokens,
                    temperature=current_temp,
                )
                pred_raw = policy_out[0] if policy_out else ""
            except Exception as e:
                logger.warning("Policy LLM call failed at round %d: %s", rnd, e)
                pred_raw = ""

            pred_element, pred_op, pred_value, pred_action_str = parse_policy_output(
                pred_raw, ctx["choices"]
            )

            # ---- judge ---- #
            judge_msgs = build_judge_messages(
                task=task,
                tree_repr=ctx["tree_repr"],
                previous_actions=previous_actions,
                candidate_choices=ctx["choices"],
                refinement_history=refinement_history,
                predicted_action_raw=pred_raw,
                trajectory_step_idx=trajectory_step_idx,
                refinement_round=rnd,
            )
            scores, feedback, judge_raw = self.judge.score(judge_msgs)

            entry = {
                "round": rnd,
                "pred_raw": pred_raw,
                "pred_element": pred_element,
                "pred_op": pred_op,
                "pred_value": pred_value,
                "pred_action_str": pred_action_str,
                "scores": scores,
                "feedback": feedback,
                "judge_raw": judge_raw,
            }
            refinement_history.append(entry)
            final_entry = entry
            previous_feedback = feedback

            if scores["total"] >= self.score_threshold:
                break

        return {
            "refinement_history": refinement_history,
            "final": final_entry,
        }

    # ----------------------- dataset driver ----------------------- #

    def evaluate_dataset(
        self,
        dataset,
        *,
        output_path: Optional[str] = None,
        name: str = "default",
    ) -> Dict[str, Any]:
        all_element_acc: List[List[Any]] = []
        all_action_f1: List[List[Any]] = []
        all_step_acc: List[List[Any]] = []
        all_final_predictions: List[List[Any]] = []
        all_trajectories: List[Dict[str, Any]] = []
        sample_to_website: Dict[str, str] = {}

        # top-k recall logging (same as ActionEvaluatorMultiChoice)
        for k in [5, 10, 20, 50]:
            recall_at_k = np.mean(
                [
                    1 if any([c.get("rank", 99999) < k for c in s["pos_candidates"]]) else 0
                    for s in dataset.data
                ]
            )
            logger.info("Recall Cap @ %d: %.4f", k, recall_at_k)
        acc_cand = np.mean(
            [
                1 if any([c.get("rank", 99999) == 0 for c in s["pos_candidates"]]) else 0
                for s in dataset.data
            ]
        )
        logger.info("Candidate generator acc (rank==0): %.4f", acc_cand)

        def _process_step(step_idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
            sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
            annotation_id = sample["annotation_id"]
            website = sample["website"]
            pos_ids_all = [c["backend_node_id"] for c in sample["pos_candidates"]]

            pos_ids_topk = [
                c["backend_node_id"]
                for c in sample["pos_candidates"]
                if c.get("rank", 99999) < self.top_k
            ]
            if not pos_ids_topk:
                return {
                    "annotation_id": annotation_id,
                    "website": website,
                    "element_acc": 0.0,
                    "action_f1": 0.0,
                    "step_acc": 0.0,
                    "final_prediction": [sample_id, "", ""],
                    "trajectory": {
                        "sample_id": sample_id,
                        "annotation_id": annotation_id,
                        "website": website,
                        "skipped": True,
                        "reason": "no positive candidate within top_k",
                        "refinement_history": [],
                    },
                }

            ctx = self._build_step_context(sample)
            one = self._refine_one_step(sample, ctx, trajectory_step_idx=step_idx)
            final = one["final"] or {}
            pred_element_id = final.get("pred_element")
            pred_action_str = final.get("pred_action_str", "")
            elem_correct = 1 if (pred_element_id is not None and pred_element_id in pos_ids_all) else 0
            f1 = calculate_f1(pred_action_str, ctx["target_action_str"])
            step_correct = 1 if (elem_correct == 1 and f1 == 1.0) else 0
            return {
                "annotation_id": annotation_id,
                "website": website,
                "element_acc": float(elem_correct),
                "action_f1": float(f1),
                "step_acc": float(step_correct),
                "final_prediction": [sample_id, pred_element_id or "", pred_action_str],
                "trajectory": {
                    "sample_id": sample_id,
                    "annotation_id": annotation_id,
                    "website": website,
                    "task": sample["confirmed_task"],
                    "previous_actions": sample.get("previous_actions", []),
                    "pos_ids": pos_ids_all,
                    "target_action_str": ctx["target_action_str"],
                    "final_round": final.get("round"),
                    "final_score": final.get("scores", {}).get("total"),
                    "refinement_history": one["refinement_history"],
                },
            }

        # ---- 按 annotation_id 分组，保留组内 step 原始顺序 ---- #
        task_groups: Dict[str, List[Any]] = collections.OrderedDict()
        global_step_indices: Dict[str, List[int]] = collections.OrderedDict()
        for global_idx, sample in enumerate(dataset.data):
            aid = sample["annotation_id"]
            if aid not in task_groups:
                task_groups[aid] = []
                global_step_indices[aid] = []
            task_groups[aid].append(sample)
            global_step_indices[aid].append(global_idx)

        def _process_task(
            annotation_id: str,
            steps: List[Any],
            step_indices: List[int],
        ) -> List[Dict[str, Any]]:
            """串行处理一个 task 的所有 steps，返回结果列表（保持 step 顺序）。"""
            return [
                _process_step(step_idx, sample)
                for step_idx, sample in zip(step_indices, steps)
            ]

        task_ids = list(task_groups.keys())
        logger.info(
            "Task-level parallelism: %d tasks, %d steps total, num_workers=%d",
            len(task_ids), len(dataset.data), self.num_workers,
        )

        with tqdm(total=len(dataset.data)) as t:
            if self.num_workers == 1:
                for aid in task_ids:
                    results = _process_task(
                        aid, task_groups[aid], global_step_indices[aid]
                    )
                    for one in results:
                        annotation_id = one["annotation_id"]
                        sample_to_website[annotation_id] = one["website"]
                        all_element_acc.append([one["element_acc"], annotation_id])
                        all_action_f1.append([one["action_f1"], annotation_id])
                        all_step_acc.append([one["step_acc"], annotation_id])
                        all_final_predictions.append(one["final_prediction"])
                        all_trajectories.append(one["trajectory"])
                        t.set_postfix(
                            element_acc=np.mean([x[0] for x in all_element_acc]),
                            action_f1=np.mean([x[0] for x in all_action_f1]),
                            step_acc=np.mean([x[0] for x in all_step_acc]),
                        )
                        t.update()
            else:
                # task 级并发：每个 future 对应一个 task 的全部 steps（组内串行）
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_aid = {
                        executor.submit(
                            _process_task,
                            aid,
                            task_groups[aid],
                            global_step_indices[aid],
                        ): aid
                        for aid in task_ids
                    }
                    # 收集结果，完成后按原始 task 顺序重排再写入
                    task_results: Dict[str, List[Dict[str, Any]]] = {}
                    for future in as_completed(future_to_aid):
                        aid = future_to_aid[future]
                        task_results[aid] = future.result()
                        t.update(len(task_results[aid]))

                # 按原始 task 顺序展平写入，保证轨迹文件有序
                for aid in task_ids:
                    for one in task_results[aid]:
                        annotation_id = one["annotation_id"]
                        sample_to_website[annotation_id] = one["website"]
                        all_element_acc.append([one["element_acc"], annotation_id])
                        all_action_f1.append([one["action_f1"], annotation_id])
                        all_step_acc.append([one["step_acc"], annotation_id])
                        all_final_predictions.append(one["final_prediction"])
                        all_trajectories.append(one["trajectory"])
                t.set_postfix(
                    element_acc=np.mean([x[0] for x in all_element_acc]),
                    action_f1=np.mean([x[0] for x in all_action_f1]),
                    step_acc=np.mean([x[0] for x in all_step_acc]),
                )

        # macro averages
        marco_element = collections.defaultdict(list)
        marco_f1 = collections.defaultdict(list)
        marco_step = collections.defaultdict(list)
        for v, a in all_element_acc:
            marco_element[a].append(v)
        for v, a in all_action_f1:
            marco_f1[a].append(v)
        for v, a in all_step_acc:
            marco_step[a].append(v)

        error_ratio: Dict[Any, float] = collections.defaultdict(int)
        acc_per_website: Dict[str, List[float]] = collections.defaultdict(list)
        for aid, xs in marco_step.items():
            acc_per_website[sample_to_website[aid]].append(float(np.mean(xs)))
            errors = len([y for y in xs if y == 0])
            if errors <= 3:
                error_ratio[errors] += 1
            else:
                error_ratio[">3"] += 1

        n_ann = max(1, len(marco_element))
        result = {
            "element_acc": float(np.mean([x[0] for x in all_element_acc])) if all_element_acc else 0.0,
            "action_f1": float(np.mean([x[0] for x in all_action_f1])) if all_action_f1 else 0.0,
            "step_acc": float(np.mean([x[0] for x in all_step_acc])) if all_step_acc else 0.0,
            "marco_element_acc": float(np.mean([np.mean(v) for v in marco_element.values()])) if marco_element else 0.0,
            "marco_action_f1": float(np.mean([np.mean(v) for v in marco_f1.values()])) if marco_f1 else 0.0,
            "marco_step_acc": float(np.mean([np.mean(v) for v in marco_step.values()])) if marco_step else 0.0,
            "error_ratio": {str(k): v / n_ann for k, v in error_ratio.items()},
            "acc_per_website": {k: (float(np.mean(v)), len(v)) for k, v in acc_per_website.items()},
            "config": {
                "max_rounds": self.max_rounds,
                "score_threshold": self.score_threshold,
                "top_k": self.top_k,
                "max_candidates": self.max_candidates,
                "num_workers": self.num_workers,
            },
        }

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, f"{name}_predictions_refine.json"), "w") as f:
                json.dump(all_final_predictions, f, ensure_ascii=False)
            with open(os.path.join(output_path, f"{name}_results_refine.json"), "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            with open(os.path.join(output_path, f"{name}_trajectories_refine.json"), "w") as f:
                json.dump(all_trajectories, f, indent=2, ensure_ascii=False)

        return result
