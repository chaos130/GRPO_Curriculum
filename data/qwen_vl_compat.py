"""Qwen-VL position_ids compatibility helpers for text-only Mind2Web rollout.

EasyR1's Qwen2/3-VL forward (`qwen2_vl_base_forward`) requires
``position_ids`` of shape ``(4, batch_size, seq_length)`` — channel 0 is the
text rope, channels 1-3 are the (T, H, W) vision rope.  For text-only inputs
the vision channels collapse to the same indices as the text channel.

Mind2Web trajectory rollout feeds the VL model with pure-text DOM prompts,
so we replicate 1D text position_ids into a 4-channel tensor when (and only
when) the loaded tokenizer/model is a Qwen-VL variant.
"""

from __future__ import annotations

from typing import Optional

import torch


def is_qwen_vl_tokenizer(tokenizer) -> bool:
    """Heuristic: check whether the tokenizer/processor belongs to Qwen-VL."""

    name = getattr(tokenizer, "name_or_path", "") or ""
    cls_name = type(tokenizer).__name__
    return "VL" in name.upper() or "Qwen2VL" in cls_name or "Qwen3VL" in cls_name


def expand_text_position_ids_for_qwen_vl(
    position_ids: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """Return position_ids in the shape EasyR1's Qwen-VL forward expects.

    If the model is not Qwen-VL, the tensor is returned unchanged.  For
    Qwen-VL with text-only inputs, the 1D text rope is replicated across the
    4 mrope channels: ``(seq_len,) -> (4, seq_len)``.
    """

    if not is_qwen_vl_tokenizer(tokenizer):
        return position_ids
    if position_ids.dim() != 1:
        return position_ids
    return position_ids.unsqueeze(0).expand(4, -1).contiguous()
