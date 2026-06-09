# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from torch.utils.data import Dataset, Sampler


class IndexedDataset(Dataset):
    """Attach stable dataset identity without changing the wrapped dataset contract."""

    def __init__(self, dataset: Dataset, task_id_key: str = "task_id") -> None:
        self.dataset = dataset
        self.task_id_key = task_id_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item["dataset_index"] = index
        item["task_id"] = str(item.get(self.task_id_key, index))
        return item


class BetaThompsonSampler(Sampler[int]):
    """Target-aware Thompson sampler over per-example Bernoulli success rates.

    Standard Thompson sampling maximizes sampled success probability and therefore
    prefers easy examples. This sampler instead prefers posterior samples close to
    ``target_success`` so it can operationalize competence-difficulty matching.
    """

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        target_success: float = 0.5,
        temperature: float = 0.15,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        uniform_mix: float = 0.1,
        seed: int = 1,
    ) -> None:
        if len(data_source) < 1:
            raise ValueError("BetaThompsonSampler requires a non-empty dataset.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive.")
        if not 0 <= target_success <= 1:
            raise ValueError("target_success must be in [0, 1].")
        if not 0 <= uniform_mix <= 1:
            raise ValueError("uniform_mix must be in [0, 1].")

        self.data_source = data_source
        self.batch_size = batch_size
        self.target_success = target_success
        self.temperature = temperature
        self.uniform_mix = uniform_mix
        self.alpha = np.full(len(data_source), prior_alpha, dtype=np.float64)
        self.beta = np.full(len(data_source), prior_beta, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        remaining = len(self)
        while remaining > 0:
            theta = self.rng.beta(self.alpha, self.beta)
            utility = np.exp(-np.abs(theta - self.target_success) / self.temperature)
            probs = utility / utility.sum()
            probs = (1.0 - self.uniform_mix) * probs + self.uniform_mix / len(self)

            sample_size = min(self.batch_size, remaining, len(self))
            indices = self.rng.choice(len(self), size=sample_size, replace=False, p=probs)
            yield from indices.tolist()
            remaining -= sample_size

    def update(self, dataset_indices: np.ndarray, successes: np.ndarray) -> None:
        indices = np.asarray(dataset_indices, dtype=np.int64)
        outcomes = np.asarray(successes, dtype=np.float64)
        if indices.shape != outcomes.shape:
            raise ValueError("dataset_indices and successes must have the same shape.")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("dataset index is outside the sampler posterior table.")

        outcomes = np.clip(outcomes, 0.0, 1.0)
        np.add.at(self.alpha, indices, outcomes)
        np.add.at(self.beta, indices, 1.0 - outcomes)

    def posterior_mean(self, dataset_indices: np.ndarray | None = None) -> np.ndarray:
        means = self.alpha / (self.alpha + self.beta)
        if dataset_indices is None:
            return means
        return means[np.asarray(dataset_indices, dtype=np.int64)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha.copy(),
            "beta": self.beta.copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.alpha = np.asarray(state_dict["alpha"], dtype=np.float64)
        self.beta = np.asarray(state_dict["beta"], dtype=np.float64)
        self.rng.bit_generator.state = state_dict["rng_state"]
