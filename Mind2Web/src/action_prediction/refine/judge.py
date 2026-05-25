"""Judge LLM wrapper with robust JSON parsing."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

SCORE_KEYS = (
    "action_validity",
    "semantic_alignment",
    "progress_contribution",
    "redundancy_loop_detection",
)


def _extract_json(text: str) -> Dict[str, Any]:
    """Locate the first balanced JSON object in ``text`` and parse it.

    Tolerates stray prose before/after the JSON (common with chat models).
    Returns {} if nothing parseable is found.
    """
    if not text:
        return {}
    # Fast path: whole string is JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    # Greedy match of the largest {...} block
    candidates = re.findall(r"\{[\s\S]*\}", text)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # Try progressively shorter prefixes of the first { ... } block
    first = re.search(r"\{[\s\S]*", text)
    if first:
        block = first.group(0)
        for end in range(len(block), 0, -1):
            if block[end - 1] != "}":
                continue
            try:
                return json.loads(block[:end])
            except json.JSONDecodeError:
                continue
    return {}


def _coerce_score(value: Any, low: int, high: int, default: int = 0) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, v))


def parse_judge_output(raw: str) -> Tuple[Dict[str, float], str]:
    """Parse a judge LLM response.

    Returns
    -------
    scores : dict with keys action_validity, semantic_alignment,
             progress_contribution, redundancy_loop_detection, total
             Score ranges: 0-2, 0-2, 0-2, 0-1; ``total`` is in [0, 7].
    feedback : str (empty if absent)
    """
    parsed = _extract_json(raw)
    scores: Dict[str, float] = {}
    scores["action_validity"] = _coerce_score(parsed.get("action_validity", 0), 0, 2)
    scores["semantic_alignment"] = _coerce_score(parsed.get("semantic_alignment", 0), 0, 2)
    scores["progress_contribution"] = _coerce_score(parsed.get("progress_contribution", 0), 0, 2)
    scores["redundancy_loop_detection"] = _coerce_score(
        parsed.get("redundancy_loop_detection", 0), 0, 1
    )
    scores["total"] = sum(scores[k] for k in SCORE_KEYS)
    feedback = parsed.get("feedback", "") or ""
    if not isinstance(feedback, str):
        feedback = str(feedback)
    return scores, feedback.strip()


class JudgeClient:
    """Thin wrapper around an `OpenaiEngine`-compatible generator."""

    def __init__(self, engine, max_new_tokens: int = 256, temperature: float = 0.0) -> None:
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def score(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, float], str, str]:
        """Return (scores, feedback, raw_text)."""
        try:
            out = self.engine.generate(
                prompt=messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            raw = out[0] if out else ""
        except Exception as e:
            logger.warning("Judge LLM call failed: %s", e)
            raw = ""
        scores, feedback = parse_judge_output(raw)
        return scores, feedback, raw
