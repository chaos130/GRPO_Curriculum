"""Hydra entrypoint for refinement-based evaluation on Mind2Web.

Run:
    python -m action_prediction.refine.run_refine \
        --config-path ../conf --config-name refine
"""
from __future__ import annotations

import logging
import os
import pathlib
import pickle
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Make sibling modules importable whether we run this file directly or via -m
_THIS = pathlib.Path(__file__).resolve()
sys.path.insert(0, _THIS.parent.parent.as_posix())
sys.path.insert(0, _THIS.parent.parent.parent.as_posix())

from dataloader import MultiChoiceDataset, get_data_split  # noqa: E402
from evaluate_llm import OpenaiEngine  # noqa: E402

from refine.evaluator import RefinementActionEvaluator  # noqa: E402

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="refine")
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.output_path, exist_ok=True)

    # ---- candidate scores (optional) ---- #
    candidate_results = None
    score_file = cfg.data.get("score_file", None)
    if score_file:
        logger.info("Loading candidate scores from %s", score_file)
        with open(score_file, "rb") as f:
            candidate_results = pickle.load(f)

    # ---- datasets ---- #
    test_datasets = {}
    for test_key, test_split_file in cfg.data.test_split_files.items():
        data = get_data_split(
            cfg.data.data_path,
            test_split_file,
            candidate_results=candidate_results,
        )
        limit = cfg.get("limit", -1)
        if limit and limit > 0:
            data = data.select(range(min(int(limit), len(data))))
        # MultiChoiceDataset wraps .data with a pass-through; we only use .data
        test_datasets[test_key] = MultiChoiceDataset(
            data,
            tokenizer=None,  # not used by the refine evaluator
            neg_ratio=0.0,
            num_candidates=cfg.refine.max_candidates,
            max_context_len=4096,
        )

    # ---- LLM engines ---- #
    policy_engine = OpenaiEngine(
        model=cfg.policy_llm,
        api_key=cfg.get("policy_api_keys", None),
        rate_limit=cfg.get("policy_rate_limit", -1),
        api_base=cfg.get("policy_api_base", None),
        api_key_env=cfg.get("policy_api_key_env", None),
        max_workers=cfg.get("llm_thread_workers", 4),
    )
    # Reuse the same class for the judge (OpenAI-compatible endpoints).
    judge_engine = OpenaiEngine(
        model=cfg.judge_llm,
        api_key=cfg.get("judge_api_keys", None),
        rate_limit=cfg.get("judge_rate_limit", -1),
        api_base=cfg.get("judge_api_base", None),
        api_key_env=cfg.get("judge_api_key_env", None),
        max_workers=cfg.get("llm_thread_workers", 4),
    )

    # ---- evaluator ---- #
    evaluator = RefinementActionEvaluator(
        policy_engine=policy_engine,
        judge_engine=judge_engine,
        max_rounds=cfg.refine.max_rounds,
        score_threshold=cfg.refine.score_threshold,
        top_k=cfg.refine.top_k,
        max_candidates=cfg.refine.max_candidates,
        previous_k=cfg.refine.previous_k,
        policy_max_new_tokens=cfg.refine.policy_max_new_tokens,
        judge_max_new_tokens=cfg.refine.judge_max_new_tokens,
        policy_temperature=cfg.refine.policy_temperature,
        policy_temperature_boost=cfg.refine.get("policy_temperature_boost", 0.0),
        judge_temperature=cfg.refine.judge_temperature,
        seed=cfg.seed,
        keep_html_brackets=cfg.refine.keep_html_brackets,
        num_workers=cfg.refine.get("num_workers", 1),
    )

    for test_key, test_dataset in test_datasets.items():
        logger.info("Start refinement evaluation for %s (%d steps)",
                    test_key, len(test_dataset.data))
        res = evaluator.evaluate_dataset(
            test_dataset,
            output_path=cfg.output_path,
            name=test_key,
        )
        logger.info("Results for %s: %s", test_key, res)


if __name__ == "__main__":
    main()
