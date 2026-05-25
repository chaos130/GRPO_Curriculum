"""Refinement-based evaluation pipeline for Mind2Web.

Implements the per-step iterative refinement loop:
    policy LLM -> predicted_action -> judge LLM -> (score, feedback)
                        ^                                      |
                        |______________ refine up to K rounds __|

final_prediction (last round) is evaluated against the ground truth.
"""

from .evaluator import RefinementActionEvaluator

__all__ = ["RefinementActionEvaluator"]
