"""Prompt builders for the refinement pipeline.

Two roles share the fixed per-step context:
    task                  : sample["confirmed_task"]
    trajectory_history    : previous ground-truth action strings
    cropped_html          : tree representation of the pruned DOM with candidates
    candidate_list        : enumerated candidates "id=N  <snippet>"
    current_step_refinement_history: list of dicts
        {round, pred_element, pred_action, pred_raw, scores, feedback}

Policy prompt additionally receives the most recent feedback so the model
sees it up front (even though it is redundant with the last history entry).

Judge prompt additionally receives refinement_round and the candidate
predicted_action to be scored.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

POLICY_SYSTEM = (
    "You are a careful web-navigation agent. Given a webpage's pruned HTML "
    "tree and a user task, pick ONE interactable element from the numbered "
    "candidate list and decide the next action.\n"
    "\n"
    "When judge feedback from prior rounds is provided:\n"
    "- If the element was wrong, select a DIFFERENT element from the candidate list.\n"
    "- If the action type was wrong, change CLICK/SELECT/TYPE accordingly.\n"
    "- If the value was wrong, correct it based on the task requirement.\n"
    "- Do NOT repeat an action that already received negative feedback.\n"
    "\n"
    "Output format (STRICT — output ONLY this, no extra text):\n"
    "Element: id=<N> <brief description copied from the candidate list>\n"
    "Action: CLICK | SELECT | TYPE\n"
    "Value: <required for SELECT/TYPE; leave empty for CLICK>\n"
    "\n"
    "If no candidate is appropriate for the task:\n"
    "Element: None\n"
    "Action: NONE\n"
    "Value:"
)
JUDGE_SYSTEM = (
    "You are an expert web-navigation action evaluator. Score the policy's "
    "predicted action by thinking step-by-step, then output a JSON object.\n"
    "\n"
    "## Evaluation Process\n"
    "\n"
    "### Step 1 — Understand the context\n"
    "Read the task, the current HTML page, and the previous ground-truth "
    "actions. Ask yourself: what has already been done? What should the "
    "current step accomplish to move the task forward?\n"
    "\n"
    "### Step 2 — Analyze the predicted action\n"
    "Identify exactly which element was selected (by its id), what action "
    "type was chosen (CLICK / SELECT / TYPE), and what value was provided. "
    "Verify whether the selected element actually appears in the candidate list.\n"
    "\n"
    "### Step 3 — Score each dimension INDEPENDENTLY\n"
    "Do not let a low score on one dimension drag down another. Each "
    "dimension should be evaluated on its own merits.\n"
    "\n"
    "**action_validity (0–2): Is the action technically correct?**\n"
    "- 2: The selected element exists in the candidate list, the action type "
    "matches the element's nature (CLICK on button/link, TYPE on input, "
    "SELECT on dropdown), and the value (if any) is appropriate.\n"
    "- 1: The element exists but the action type is slightly mismatched, the "
    "value has minor issues, or the element is ambiguous.\n"
    "- 0: Element not found in candidates, or action type is clearly wrong "
    "(e.g., CLICK when TYPE is needed), or value is missing for TYPE/SELECT.\n"
    "\n"
    "**semantic_alignment (0–2): Does the action serve the user's task?**\n"
    "- 2: The action directly and specifically advances the stated task at "
    "this particular step.\n"
    "- 1: The action is plausibly related to the task but is not clearly the "
    "right next move, or it addresses the task too broadly.\n"
    "- 0: The action is unrelated to the task, contradictory, or clearly "
    "off-goal.\n"
    "\n"
    "**progress_contribution (0–2): Does the action make meaningful progress?**\n"
    "- 2: This action is a clear, necessary step forward — it would bring "
    "the user noticeably closer to completing the task.\n"
    "- 1: The action is not harmful but does not clearly advance the task, "
    "or its contribution is marginal.\n"
    "- 0: The action would undo progress, create a dead end, or waste a step.\n"
    "\n"
    "**redundancy_loop_detection (0–1): Is this a new, meaningful attempt?**\n"
    "- 1: The action is meaningfully different from prior attempts in this "
    "step, or this is the first attempt.\n"
    "- 0: The action substantially repeats a prior failed attempt without "
    "addressing the judge's feedback.\n"
    "\n"
    "### Step 4 — Constructive feedback\n"
    "Provide specific, actionable feedback for the policy:\n"
    "- If the element is wrong, describe what KIND of element to look for.\n"
    "- If the action type is wrong, explain why and suggest the correct type.\n"
    "- If the value is wrong, indicate what the value should contain.\n"
    "- Keep feedback to 2–3 concise sentences.\n"
    "\n"
    "## Output Format (JSON ONLY, no extra text)\n"
    "{\n"
    "  \"analysis\": \"<your brief reasoning, 2–4 sentences>\",\n"
    "  \"action_validity\": <int 0-2>,\n"
    "  \"semantic_alignment\": <int 0-2>,\n"
    "  \"progress_contribution\": <int 0-2>,\n"
    "  \"redundancy_loop_detection\": <int 0-1>,\n"
    "  \"feedback\": \"<specific, actionable advice>\"\n"
    "}\n"
    "\n"
    "The caller computes total_score = sum of the four dimensions (range 0–7).\n"
)


def _format_trajectory_history(previous_actions: List[str], k: int = 10) -> str:
    if not previous_actions:
        return "None"
    keep = previous_actions[-k:]
    return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(keep))


def _format_candidate_list(choices: List[List[str]]) -> str:
    """choices is a list of [backend_node_id, short_repr] from format_input_generation.
    The short_repr already contains `id=<N>` when keep_html_brackets matches the
    mode used to build it; we prepend an index for easier reference.
    """
    lines = []
    for idx, choice in enumerate(choices):
        snippet = choice[1] if len(choice) > 1 else ""
        lines.append(f"[{idx}] {snippet}")
    return "\n".join(lines) if lines else "(no candidates)"


def _format_refinement_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "None (this is refinement round 1)"
    blocks = []
    for entry in history:
        scores = entry.get("scores", {}) or {}
        score_line = ", ".join(
            f"{k}={v}" for k, v in scores.items() if k not in {"mean", "total"}
        )
        total = scores.get("total")
        total_str = f" | total={total}" if isinstance(total, (int, float)) else ""
        blocks.append(
            f"--- round {entry['round']} ---\n"
            f"Predicted:\n{entry.get('pred_raw', '').strip()}\n"
            f"Judge scores: {score_line}{total_str}\n"
            f"Judge feedback: {entry.get('feedback', '').strip()}"
        )
    return "\n\n".join(blocks)


def build_policy_messages(
    *,
    task: str,
    tree_repr: str,
    previous_actions: List[str],
    candidate_choices: List[List[str]],
    refinement_history: List[Dict[str, Any]],
    previous_feedback: Optional[str],
    trajectory_step_idx: int,
    refinement_round: int,
) -> List[Dict[str, str]]:
    user = (
        f"# Task\n{task}\n\n"
        f"# Trajectory step index\n{trajectory_step_idx}\n\n"
        f"# Refinement round\n{refinement_round}\n\n"
        f"# Cropped HTML (pruned DOM with candidates)\n'''\n{tree_repr}\n'''\n\n"
        f"# Ground-truth previous actions (trajectory history)\n"
        f"{_format_trajectory_history(previous_actions)}\n\n"
        f"# Candidate elements (you MUST pick one of these ids, or 'None')\n"
        f"{_format_candidate_list(candidate_choices)}\n\n"
        f"# Prior attempts in this step (refinement history)\n"
        f"{_format_refinement_history(refinement_history)}\n\n"
        f"# Most recent judge feedback to address (if any)\n"
        f"{previous_feedback.strip() if previous_feedback else 'N/A'}\n\n"
        f"# Your output\n"
        f"Follow the STRICT format from the system message."
    )
    return [
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_judge_messages(
    *,
    task: str,
    tree_repr: str,
    previous_actions: List[str],
    candidate_choices: List[List[str]],
    refinement_history: List[Dict[str, Any]],
    predicted_action_raw: str,
    trajectory_step_idx: int,
    refinement_round: int,
) -> List[Dict[str, str]]:
    user = (
        f"# Task\n{task}\n\n"
        f"# Trajectory step index\n{trajectory_step_idx}\n\n"
        f"# Refinement round being judged\n{refinement_round}\n\n"
        f"# Cropped HTML (pruned DOM)\n'''\n{tree_repr}\n'''\n\n"
        f"# Ground-truth previous actions\n"
        f"{_format_trajectory_history(previous_actions)}\n\n"
        f"# Candidate elements visible to the policy\n"
        f"{_format_candidate_list(candidate_choices)}\n\n"
        f"# Prior refinement history (earlier rounds of THIS step)\n"
        f"{_format_refinement_history(refinement_history)}\n\n"
        f"# Policy prediction to score (this round)\n"
        f"{predicted_action_raw.strip()}\n\n"
        f"# Your output\nReturn ONLY the JSON object specified in the system message."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
