# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


_GRPO_CURRICULUM_ROOT = Path(__file__).resolve().parents[3]
if _GRPO_CURRICULUM_ROOT.as_posix() not in sys.path:
    # EasyR1 is launched from its own directory; add the parent framework layer
    # so dataset adapters under GRPO_Curriculum/data are importable.
    sys.path.insert(0, _GRPO_CURRICULUM_ROOT.as_posix())


def create_dataset(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    split: str,
):
    """Create the dataset for a train/validation split.

    The default branch is exactly EasyR1's original RLHFDataset.  Non-default
    branches are thin adapters that still return the same batch contract.
    """

    data_path = config.train_files if split == "train" else config.val_files
    if config.dataset_type == "rlhf":
        return RLHFDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            format_prompt_kwargs=config.format_prompt_kwargs,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        )

    if config.dataset_type == "mind2web_trajectory":
        from data.adapters.mind2web_trajectory import Mind2WebTrajectoryDataset

        dataset_kwargs = dict(config.dataset_kwargs)
        data_root = dataset_kwargs.pop("data_path")
        return Mind2WebTrajectoryDataset(
            data_path=data_root,
            split_file=data_path,
            tokenizer=tokenizer,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            **dataset_kwargs,
        )

    raise ValueError(f"Unsupported dataset_type: {config.dataset_type}")


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = create_dataset(config, tokenizer, processor, split="train")
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = create_dataset(config, tokenizer, processor, split="val")

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
