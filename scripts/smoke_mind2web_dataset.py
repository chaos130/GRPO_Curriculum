#!/usr/bin/env python3
"""CPU smoke test: Mind2Web trajectory dataset + state prompts (no GPU / vLLM).

Framework script — backends: Mind2Web (data) + transformers tokenizer only.
Run from repo root: python scripts/smoke_mind2web_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from data.adapters.mind2web_trajectory import Mind2WebTrajectoryDataset


def _default_paths() -> dict[str, str]:
    """Docker (/workspace/*) vs host (/mnt/sda/Xml/workplace/*)."""
    if Path("/workspace/model").is_dir() and Path("/workspace/data").is_dir():
        data_root = Path("/workspace/data/Mind2Web")
        model = Path("/workspace/model/Qwen/Qwen2.5-VL-3B-Instruct")
    else:
        data_root = Path("/mnt/sda/Xml/workplace/data/Mind2Web")
        model = Path("/mnt/sda/Xml/workplace/model/Qwen/Qwen2.5-VL-3B-Instruct")
    return {
        "data_path": str(data_root / "data"),
        "split_file": str(data_root / "data/test_task/test_task_0.json"),
        "score_file": str(data_root / "src/scores_all_data.pkl"),
        "model_path": str(model),
    }


def main() -> None:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Smoke-test Mind2WebTrajectoryDataset")
    parser.add_argument(
        "--data-path",
        default=defaults["data_path"],
        help="Mind2Web data root (contains train/, test_task/, ...)",
    )
    parser.add_argument(
        "--split-file",
        default=defaults["split_file"],
        help="JSON split file for load_dataset",
    )
    parser.add_argument(
        "--score-file",
        default=defaults["score_file"],
    )
    parser.add_argument("--model-path", default=defaults["model_path"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Building Mind2WebTrajectoryDataset ...")
    dataset = Mind2WebTrajectoryDataset(
        data_path=args.data_path,
        split_file=args.split_file,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        candidate_source="ranked",
        score_file=args.score_file,
        top_k=50,
        max_candidates=20,
        previous_k=5,
        task_filter="none",
    )
    print(f"Dataset size: {len(dataset)} tasks")

    sample = dataset[args.index]
    traj = sample["trajectory_data"]
    print("\n=== Task summary ===")
    print(f"  website: {traj['website']}")
    print(f"  task: {traj['confirmed_task'][:120]}...")
    print(f"  num_steps: {len(traj['steps'])}")

    step0 = traj["steps"][0]
    state_prompt = step0["state_prompt"]
    print("\n=== Step 0 state_prompt (first 800 chars) ===")
    print(state_prompt[:800])
    print(f"\n  state_prompt chars: {len(state_prompt)}")
    print(f"  seq_target:\n{step0['seq_target'][:400]}")

    print(f"\n  ground_truth (truncated): {sample['ground_truth'][:200]}...")

    preview = {
        "website": traj["website"],
        "confirmed_task": traj["confirmed_task"],
        "num_steps": len(traj["steps"]),
        "step0_state_prompt_head": state_prompt[:500],
        "step0_seq_target": step0["seq_target"],
        "gold_trajectory_len": len(traj["gold_trajectory"]),
    }
    out = REPO_ROOT / "scripts" / "smoke_mind2web_preview.json"
    out.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote preview to {out}")
    print("OK: dataset + prompt construction passed.")


if __name__ == "__main__":
    main()
