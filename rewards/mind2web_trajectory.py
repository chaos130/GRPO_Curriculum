"""Smoke-test reward for Mind2Web trajectory rollout.

This is intentionally small: it only checks whether a generated step action is
parseable.  The real trajectory-level reward should later consume
`trajectory_id`, `predicted_trajectory`, and per-step metadata from the rollout
adapter.  Keeping this placeholder lets us verify data loading, rollout, and
policy-update plumbing before designing reward semantics.
"""

from __future__ import annotations

import re
from typing import Any


REWARD_NAME = "mind2web_trajectory_smoke"
REWARD_TYPE = "batch"

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(CLICK|TYPE|SELECT|NONE)", re.IGNORECASE)
_ELEMENT_RE = re.compile(r"Element:\s*(.+)", re.IGNORECASE)


def _extract_answer(response: str) -> str:
    """Extract the answer block when EasyR1-style formatting is present."""

    match = _ANSWER_RE.search(response or "")
    return match.group(1).strip() if match else (response or "").strip()


def _format_score(response: str) -> float:
    """Give a small scalar score for parseable next-action text."""

    answer = _extract_answer(response)
    score = 0.0
    if _ELEMENT_RE.search(answer):
        score += 0.4
    if _ACTION_RE.search(answer):
        score += 0.6
    return score


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Return one parseability reward per generated step action."""

    outputs = []
    for reward_input in reward_inputs:
        score = _format_score(str(reward_input.get("response", "")))
        outputs.append(
            {
                "overall": score,
                "format": score,
                "accuracy": 0.0,
            }
        )
    return outputs

