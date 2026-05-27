"""Mind2Web trajectory step reward.

Two dimensions:
  * format  — does the response follow the strict POLICY_SYSTEM output shape?
  * answer  — is the predicted action correct, decomposed into three parts:
        action (weight 0.3) | id (weight 0.4) | value (weight 0.3)

The reward consumes per-row `step_data` (attached by the Mind2Web rollout
adapter) for the gold action.  When `step_data` is unavailable (e.g. running
this reward outside the Mind2Web pipeline) we fall back to format-only scoring.

Overall = 0.5 * format + 0.5 * answer (both already in [0, 1]).
"""

from __future__ import annotations

import re
from typing import Any, Optional

REWARD_NAME = "mind2web_trajectory_step"
REWARD_TYPE = "batch"

# Weights for the three answer sub-components (must sum to 1.0).
W_ACTION = 0.3
W_ID = 0.4
W_VALUE = 0.3

# Weights for combining the two top-level dimensions.
W_FORMAT = 0.5
W_ANSWER = 0.5

_VALID_ACTIONS = {"CLICK", "SELECT", "TYPE", "NONE"}

# Patterns shared by gold (`seq_target`) and predicted response.
_ELEMENT_RE = re.compile(r"Element:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z]+)", re.IGNORECASE)
_VALUE_RE = re.compile(r"Value:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_ID_RE = re.compile(r"id\s*=\s*(\d+)")


def _parse_response(text: str) -> dict[str, Optional[str]]:
    text = text or ""
    element_match = _ELEMENT_RE.search(text)
    action_match = _ACTION_RE.search(text)
    value_match = _VALUE_RE.search(text)

    element_text = element_match.group(1).strip() if element_match else None
    action = action_match.group(1).strip().upper() if action_match else None
    value = value_match.group(1).strip() if value_match else None

    element_id = None
    if element_text:
        id_match = _ID_RE.search(element_text)
        if id_match:
            element_id = id_match.group(1)
        elif element_text.lower().startswith("none"):
            element_id = "NONE"

    return {
        "element_text": element_text,
        "element_id": element_id,
        "action": action,
        "value": value,
    }


def _gold_from_seq_target(seq_target: Optional[str]) -> dict[str, Optional[str]]:
    """seq_target either looks like the response format, or literally 'None'."""

    if not seq_target or seq_target.strip().lower() == "none":
        return {"element_id": "NONE", "action": "NONE", "value": None}
    parsed = _parse_response(seq_target)
    parsed.setdefault("element_id", None)
    parsed.setdefault("action", None)
    return parsed


def _format_score(parsed: dict[str, Optional[str]], raw_response: str) -> float:
    """1.0 if the response strictly matches the policy output schema."""

    if not raw_response:
        return 0.0

    action = parsed["action"]
    element_id = parsed["element_id"]

    if action not in _VALID_ACTIONS:
        return 0.0
    if element_id is None:
        return 0.0

    # CLICK/NONE: Value line must be absent OR empty.
    # SELECT/TYPE: Value line must be present and non-empty.
    has_value_field = parsed["value"] is not None and parsed["value"] != ""
    if action in {"CLICK", "NONE"}:
        if parsed["value"] not in (None, ""):
            return 0.5  # mostly correct but extra Value content
    else:  # SELECT / TYPE
        if not has_value_field:
            return 0.5

    # Penalise responses that contain extra prose outside the three required
    # lines (e.g. "Thought: ...").  We approximate "extra text" as any line
    # that is not blank and not one of the three expected prefixes.
    for line in raw_response.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if not (lower.startswith("element:") or lower.startswith("action:") or lower.startswith("value:")):
            return 0.7
    return 1.0


def _normalize_value(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _answer_score(
    pred: dict[str, Optional[str]],
    gold: dict[str, Optional[str]],
) -> dict[str, float]:
    gold_action = gold.get("action")
    gold_id = gold.get("element_id")
    gold_value = gold.get("value")

    action_hit = 1.0 if pred["action"] and gold_action and pred["action"] == gold_action else 0.0
    id_hit = 1.0 if pred["element_id"] and gold_id and pred["element_id"] == gold_id else 0.0

    # Value only matters for TYPE / SELECT.  For CLICK / NONE the gold has no
    # value; we credit the prediction if it also has no value.
    if gold_action in {"CLICK", "NONE"} or not gold_action:
        value_hit = 1.0 if not pred["value"] else 0.0
    else:
        value_hit = 1.0 if _normalize_value(pred["value"]) == _normalize_value(gold_value) else 0.0

    answer = W_ACTION * action_hit + W_ID * id_hit + W_VALUE * value_hit
    return {
        "action_hit": action_hit,
        "id_hit": id_hit,
        "value_hit": value_hit,
        "answer": answer,
    }


def _extract_gold(reward_input: dict[str, Any]) -> Optional[dict[str, Optional[str]]]:
    step_data = reward_input.get("step_data")
    if isinstance(step_data, dict):
        seq_target = step_data.get("seq_target")
        if seq_target is not None:
            return _gold_from_seq_target(str(seq_target))
    return None


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compute per-step reward: format (0..1) + answer (0..1) → overall."""

    outputs = []
    for reward_input in reward_inputs:
        response = str(reward_input.get("response", ""))
        pred = _parse_response(response)
        format_score = _format_score(pred, response)

        gold = _extract_gold(reward_input)
        if gold is None:
            outputs.append(
                {
                    "overall": W_FORMAT * format_score,
                    "format": format_score,
                    "answer": 0.0,
                    "action_hit": 0.0,
                    "id_hit": 0.0,
                    "value_hit": 0.0,
                }
            )
            continue

        ans = _answer_score(pred, gold)
        overall = W_FORMAT * format_score + W_ANSWER * ans["answer"]
        outputs.append(
            {
                "overall": overall,
                "format": format_score,
                "answer": ans["answer"],
                "action_hit": ans["action_hit"],
                "id_hit": ans["id_hit"],
                "value_hit": ans["value_hit"],
            }
        )
    return outputs
