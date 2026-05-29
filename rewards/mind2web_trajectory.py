"""Mind2Web trajectory step reward.

Two dimensions:
  * format  — continuous structural score for POLICY_SYSTEM layout (line-anchored).
  * answer  — semantic correctness vs gold: action | id | value.

Format deliberately avoids Value *content* / presence rules so we do not double-penalise
with ``value_hit`` in the answer head.  Continuous format scores help GRPO: when two
rollouts tie on answer hits, small structural differences still spread ``overall`` and
keep group std > 0 (plateau-style 0.5/0.7/1.0 buckets often zero out advantage).

Overall = 0.5 * format + 0.5 * answer (both in [0, 1]).
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

# Line-anchored patterns aligned with POLICY_SYSTEM's strict three-line output.
# MULTILINE + ^ avoids matching "Element:" inside free text on another line.
_LINE_FLAGS = re.IGNORECASE | re.MULTILINE
_ELEMENT_LINE_RE = re.compile(r"^Element:\s*(.+)\s*$", _LINE_FLAGS)
_ACTION_LINE_RE = re.compile(r"^Action:\s*(\S+)\s*$", _LINE_FLAGS)
_VALUE_LINE_RE = re.compile(r"^Value:\s*(.*)\s*$", _LINE_FLAGS)
_SCHEMA_PREFIX_RE = re.compile(r"^(?:Element|Action|Value):", _LINE_FLAGS)
_ID_RE = re.compile(r"id\s*=\s*(\d+)", re.IGNORECASE)

# Linear format penalties (sum capped via clamp).  Tuned so minor diffs break GRPO ties.
_FMT_PENALTY_MISSING_ELEMENT = 0.35
_FMT_PENALTY_MISSING_ACTION = 0.25
_FMT_PENALTY_INVALID_ACTION = 0.20
_FMT_PENALTY_MISSING_ELEMENT_ID = 0.25
_FMT_PENALTY_EXTRA_LINE = 0.08


def _parse_response(text: str) -> dict[str, Optional[str]]:
    """Parse the first line-anchored Element / Action / Value fields."""

    text = text or ""
    element_match = _ELEMENT_LINE_RE.search(text)
    action_match = _ACTION_LINE_RE.search(text)
    value_match = _VALUE_LINE_RE.search(text)

    element_text = element_match.group(1).strip() if element_match else None
    action = action_match.group(1).strip().upper() if action_match else None
    # Value line may be legally absent for CLICK; keep None vs "" distinct for value_hit.
    value = value_match.group(1).strip() if value_match is not None else None

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


def _count_extra_prose_lines(raw_response: str) -> int:
    """Count non-empty lines that are not Element:/Action:/Value: at line start."""

    extra = 0
    for line in raw_response.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SCHEMA_PREFIX_RE.match(stripped):
            continue
        extra += 1
    return extra


def _format_score(parsed: dict[str, Optional[str]], raw_response: str) -> float:
    """Continuous structural score in [0, 1]; starts at 1.0 and subtracts per missing field.

    Does *not* judge Value presence or content — that belongs to ``value_hit`` so CLICK/SELECT
    value mistakes are not penalised twice (old plateau returned 0.5 in format *and* 0 in value_hit).
    """

    if not raw_response or not raw_response.strip():
        return 0.0

    score = 1.0
    action = parsed["action"]
    element_id = parsed["element_id"]

    if parsed["element_text"] is None:
        score -= _FMT_PENALTY_MISSING_ELEMENT
    if action is None:
        score -= _FMT_PENALTY_MISSING_ACTION
    elif action not in _VALID_ACTIONS:
        score -= _FMT_PENALTY_INVALID_ACTION
    if element_id is None:
        score -= _FMT_PENALTY_MISSING_ELEMENT_ID

    score -= _FMT_PENALTY_EXTRA_LINE * _count_extra_prose_lines(raw_response)

    return max(0.0, min(1.0, score))


def _normalize_value(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _answer_score(
    pred: dict[str, Optional[str]],
    gold: dict[str, Optional[str]],
) -> dict[str, float]:
    """Semantic match vs gold.  Value rules live here only (not in format)."""

    gold_action = gold.get("action")
    gold_id = gold.get("element_id")
    gold_value = gold.get("value")

    action_hit = 1.0 if pred["action"] and gold_action and pred["action"] == gold_action else 0.0
    id_hit = 1.0 if pred["element_id"] and gold_id and pred["element_id"] == gold_id else 0.0

    # Value: CLICK/NONE gold should have no Value line; SELECT/TYPE must match gold text.
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
