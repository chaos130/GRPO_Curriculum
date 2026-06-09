# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..protocol import DataProto
from .beta_thompson_sampler import BetaThompsonSampler
from .ray_trainer import RayPPOTrainer


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_ANDROID_FALLBACK_RE = re.compile(r"[012]")


def _extract_answer(text: Any) -> str:
    value = str(text).strip()
    match = _ANSWER_RE.search(value)
    if match:
        return match.group(1).strip()
    if value in {"0", "1", "2"}:
        return value
    match = _ANDROID_FALLBACK_RE.search(value)
    return match.group(0) if match else value


class BetaThompsonRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer that updates the dataloader's Beta posterior after rollout.

    The posterior observes binary final-answer correctness only. Dense overall
    rewards and LLM-judge rankings are deliberately excluded from the Bernoulli
    update so the posterior keeps a clear success-probability interpretation.
    """

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = super()._make_batch_data(metrics)
        sampler = self.train_dataloader.sampler
        if not isinstance(sampler, BetaThompsonSampler):
            return batch

        required = {"dataset_index", "ground_truth"}
        missing = required.difference(batch.non_tensor_batch)
        if missing:
            raise KeyError(f"Beta Thompson sampling requires batch fields: {sorted(missing)}")

        response_ids = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        ground_truths = batch.non_tensor_batch["ground_truth"]
        indices = np.asarray(batch.non_tensor_batch["dataset_index"], dtype=np.int64)

        successes = []
        for row in range(len(batch)):
            valid_length = int(response_mask[row].sum().item())
            response = self.tokenizer.decode(response_ids[row][:valid_length], skip_special_tokens=True)
            successes.append(float(_extract_answer(response) == _extract_answer(ground_truths[row])))

        outcomes = np.asarray(successes, dtype=np.float64)
        sampler.update(indices, outcomes)
        selected_means = sampler.posterior_mean(indices)
        all_means = sampler.posterior_mean()

        metrics.update(
            {
                "beta_ts/batch_success_rate": float(outcomes.mean()),
                "beta_ts/selected_posterior_mean": float(selected_means.mean()),
                "beta_ts/posterior_mean": float(all_means.mean()),
                "beta_ts/posterior_min": float(all_means.min()),
                "beta_ts/posterior_max": float(all_means.max()),
            }
        )
        return batch
