#!/usr/bin/env python3
"""Print a compact s,a,s1,a1 timeline from Mind2Web rollout trajectory JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _print_legacy_groups(payload: dict) -> None:
    print("Detected legacy flat format (groups). Re-run rollout after the latest dump update.")
    for group in payload.get("groups", []):
        print(f"\n=== group {group.get('group_id')} ===")
        for rollout in group.get("rollouts", []):
            print(f"  rollout {rollout.get('rollout_index')}: {rollout.get('response', '')[:120]}")


def _extract_previous_actions(prompt_text: str) -> list[str]:
    marker = "Previous actions:\n"
    if marker not in prompt_text:
        return []
    start = prompt_text.index(marker) + len(marker)
    end = prompt_text.find("What should be the next action?", start)
    block = prompt_text[start:end] if end != -1 else prompt_text[start:]
    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
    if len(lines) == 1 and lines[0].lower() == "none":
        return []
    return lines


def _extract_task_line(prompt_text: str) -> str:
    marker = "Task:"
    if marker not in prompt_text:
        return ""
    start = prompt_text.index(marker) + len(marker)
    end = prompt_text.find("\nPrevious actions:", start)
    if end == -1:
        end = prompt_text.find("\nWhat should be the next action?", start)
    return prompt_text[start:end].strip() if end != -1 else prompt_text[start:].strip()


def _extract_dom_head(prompt_text: str, n_chars: int = 800) -> str:
    """Return the first n_chars of the DOM tree segment of the prompt."""
    # state_prompt = "system\n...assistant\nuser\n<DOM>\nBased on the HTML..."
    start = 0
    user_marker = "\nuser\n"
    if user_marker in prompt_text:
        start = prompt_text.index(user_marker) + len(user_marker)
    end = prompt_text.find("\nBased on the HTML", start)
    dom = prompt_text[start:end] if end != -1 else prompt_text[start:]
    dom = dom.strip().replace("\n", " ")
    if len(dom) > n_chars:
        dom = dom[:n_chars] + "..."
    return dom


def _step_timeline_from_steps(steps: list[dict]) -> list[dict]:
    timeline = []
    for step in steps:
        gold = step.get("gold") or {}
        reward = step.get("reward") or {}
        prompt_text = step.get("state_prompt", "")
        timeline.append(
            {
                "step_index": step.get("step_index"),
                "task": _extract_task_line(prompt_text),
                "dom_head": _extract_dom_head(prompt_text),
                "previous_actions": _extract_previous_actions(prompt_text),
                "predicted_action": step.get("response"),
                "gold_action": gold.get("target_action"),
                "gold_seq_target": gold.get("seq_target"),
                "reward_overall": reward.get("overall", step.get("overall_score")),
                "advantage_mean": step.get("advantage_mean"),
            }
        )
    return timeline


def _print_timeline(timeline: list[dict], indent: str = "    ") -> None:
    for item in timeline:
        prev = item.get("previous_actions") or []
        prev_text = "None" if not prev else " | ".join(prev)
        print(f"{indent}step {item['step_index']}:")
        if item.get("task"):
            print(f"{indent}  s (task): {item['task']}")
        if item.get("dom_head"):
            print(f"{indent}  s (DOM): {item['dom_head']}")
        print(f"{indent}  s (prev): {prev_text}")
        print(f"{indent}  a (pred): {item.get('predicted_action')}")
        print(f"{indent}  a* (gold): {item.get('gold_action')}")
        reward = item.get("reward_overall")
        if reward is not None:
            print(f"{indent}  reward: {reward}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_path",
        type=Path,
        default=Path(
            "EasyR1/checkpoints/grpo_curriculum/mind2web_trajectory_debug_rollout/"
            "rollout_trajectories/step_0001.json"
        ),
        nargs="?",
    )
    args = parser.parse_args()
    path = args.json_path.resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    payload = _load(path)
    print(f"file: {path}")
    print(
        f"global_step={payload.get('global_step')} "
        f"rollout_n={payload.get('rollout_n')} "
        f"num_tasks={payload.get('num_tasks', payload.get('num_prompts'))}"
    )

    tasks = payload.get("tasks")
    if not tasks:
        _print_legacy_groups(payload)
        return

    for task in tasks:
        print(f"\n=== task {task.get('task_index')} | uid={task.get('task_uid')} ===")
        instruction = task.get("task_instruction")
        if not instruction and task.get("trajectories"):
            first_steps = task["trajectories"][0].get("steps") or []
            if first_steps and first_steps[0].get("state_prompt"):
                prompt = first_steps[0]["state_prompt"]
                if "Task:" in prompt:
                    start = prompt.index("Task:") + len("Task:")
                    end = prompt.find("\nPrevious actions:", start)
                    instruction = prompt[start:end].strip() if end != -1 else prompt[start:].strip()
        if instruction:
            print(f"instruction: {instruction}")
        for traj in task.get("trajectories", []):
            print(
                f"\n  --- trajectory {traj.get('rollout_index')} "
                f"({traj.get('trajectory_id')}) ---"
            )
            timeline = traj.get("timeline")
            if not timeline:
                timeline = _step_timeline_from_steps(traj.get("steps", []))
            _print_timeline(timeline)


if __name__ == "__main__":
    main()
