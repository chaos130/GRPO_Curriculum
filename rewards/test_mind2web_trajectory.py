"""Tests for Mind2Web trajectory step reward."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


_REWARD_PATH = Path(__file__).resolve().parent / "mind2web_trajectory.py"
_spec = importlib.util.spec_from_file_location("mind2web_trajectory_reward", _REWARD_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_score = _mod.compute_score
_parse_response = _mod._parse_response
_format_score = _mod._format_score


def _step_input(response: str, seq_target: str) -> dict:
    return {
        "response": response,
        "step_data": {"seq_target": seq_target},
    }


GOLD_CLICK = (
    "Element: (<button> id=12 Search flights)\n"
    "Action: CLICK\n"
)

GOLD_TYPE = (
    "Element: (<input> id=5 Date field)\n"
    "Action: TYPE\n"
    "Value: 2024-01-01\n"
)


class TestParseResponse(unittest.TestCase):
    def test_line_anchored_fields_only(self):
        text = (
            "Thought: ignore me\n"
            "Element: (<button> id=3 Go)\n"
            "Action: CLICK\n"
        )
        parsed = _parse_response(text)
        self.assertEqual(parsed["element_id"], "3")
        self.assertEqual(parsed["action"], "CLICK")

    def test_inline_element_not_matched_without_line_start(self):
        text = "Note Element: (<button> id=9 Fake)\nAction: CLICK\n"
        parsed = _parse_response(text)
        self.assertIsNone(parsed["element_text"])
        self.assertEqual(parsed["action"], "CLICK")


class TestFormatScore(unittest.TestCase):
    def test_perfect_click_is_one(self):
        parsed = _parse_response(GOLD_CLICK)
        self.assertEqual(_format_score(parsed, GOLD_CLICK), 1.0)

    def test_continuous_penalty_for_extra_prose(self):
        clean = GOLD_CLICK
        with_thought = "Thought: planning\n" + GOLD_CLICK
        s_clean = _format_score(_parse_response(clean), clean)
        s_thought = _format_score(_parse_response(with_thought), with_thought)
        self.assertLess(s_thought, s_clean)
        self.assertAlmostEqual(s_thought, 1.0 - _mod._FMT_PENALTY_EXTRA_LINE)

    def test_select_missing_value_not_penalised_in_format(self):
        missing_value = (
            "Element: (<input> id=5 Date field)\n"
            "Action: TYPE\n"
        )
        parsed = _parse_response(missing_value)
        self.assertEqual(_format_score(parsed, missing_value), 1.0)


class TestComputeScore(unittest.TestCase):
    def test_two_rollouts_differ_on_format_when_answer_ties(self):
        gold = GOLD_CLICK
        r1 = GOLD_CLICK
        r2 = "Thought: hmm\n" + GOLD_CLICK
        scores = compute_score([_step_input(r1, gold), _step_input(r2, gold)])
        self.assertEqual(scores[0]["format"], 1.0)
        self.assertLess(scores[1]["format"], 1.0)

    def test_select_missing_value_single_penalty_via_value_hit(self):
        pred = (
            "Element: (<input> id=5 Date field)\n"
            "Action: TYPE\n"
        )
        out = compute_score([_step_input(pred, GOLD_TYPE)])[0]
        self.assertEqual(out["format"], 1.0)
        self.assertEqual(out["value_hit"], 0.0)
        self.assertAlmostEqual(out["answer"], 0.7)

    def test_continuous_format_breaks_partial_tie(self):
        gold = GOLD_CLICK
        r1 = (
            "Element: (<button> id=99 Search flights)\n"
            "Action: CLICK\n"
        )
        r2 = (
            "Thought: wrong id\n"
            "Element: (<button> id=99 Search flights)\n"
            "Action: CLICK\n"
        )
        s1, s2 = compute_score([_step_input(r1, gold), _step_input(r2, gold)])
        self.assertEqual(s1["id_hit"], 0.0)
        self.assertEqual(s2["id_hit"], 0.0)
        self.assertNotEqual(s1["overall"], s2["overall"])
        self.assertLess(s2["overall"], s1["overall"])


if __name__ == "__main__":
    unittest.main()
