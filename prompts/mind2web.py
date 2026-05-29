"""Mind2Web prompt/state builders for offline trajectory rollout.

Mind2Web stores a task as a fixed sequence of offline webpage states.  In the
original SFT code each state prompt is built from a pruned DOM tree plus a
natural-language task/history instruction.  This module keeps that construction
in one place so both dataset loading and trajectory rollout use the same state
definition.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import lxml.etree


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MIND2WEB_SRC = _REPO_ROOT / "Mind2Web" / "src"
if _MIND2WEB_SRC.as_posix() not in sys.path:
    # Mind2Web's original files import data_utils as a top-level package.
    sys.path.insert(0, _MIND2WEB_SRC.as_posix())

from data_utils.dom_utils import get_tree_repr, prune_tree  # type: ignore  # noqa: E402

POLICY_SYSTEM = (
    "You are a careful web-navigation agent. Given a webpage's pruned HTML "
    "tree and a user task, pick ONE interactable element from the tree (each "
    "candidate is annotated with id=<N>) and decide the next action.\n"
    "\n"
    "Your output must be no more than xxx tokens.\n"
    "\n"
    "Output format (STRICT — output ONLY this, no extra text, no thoughts):\n"
    "Element: (<tag> id=<N> <short description copied from the tree>)\n"
    "Action: CLICK | SELECT | TYPE\n"
    "Value: <required for SELECT/TYPE; omit the Value line entirely for CLICK>\n"
    "\n"
    "If no element in the tree is appropriate for this step:\n"
    "Element: None\n"
    "Action: NONE\n"
)
# POLICY_SYSTEM = (
#     "You are a careful web-navigation agent. Given a webpage's pruned HTML "
#     "tree and a user task, pick ONE interactable element from the tree (each "
#     "candidate is annotated with id=<N>) and decide the next action.\n"
#     "\n"
#     "Output format (STRICT — output ONLY this, no extra text, no thoughts):\n"
#     "Element: (<tag> id=<N> <short description copied from the tree>)\n"
#     "Action: CLICK | SELECT | TYPE\n"
#     "Value: <required for SELECT/TYPE; omit the Value line entirely for CLICK>\n"
#     "\n"
#     "If no element in the tree is appropriate for this step:\n"
#     "Element: None\n"
#     "Action: NONE\n"
# )


def build_seq_input(confirmed_task: str, previous_actions: list[str], previous_k: int = 5) -> str:
    """Build the textual task/history instruction used by Mind2Web SFT.

    This intentionally mirrors `format_input_generation` in Mind2Web's
    action_prediction dataloader so that state S has the same semantics as the
    benchmark's supervised fine-tuning prompt.
    """

    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {confirmed_task}\n"
        "Previous actions:\n"
    )
    if len(previous_actions) > 0:
        for action in previous_actions[-previous_k:]:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"

    # Keep the original wording, including the missing space after '?'.
    seq_input += (
        "What should be the next action?"
        "Please select the element to interact with, and the action to perform along with the value to type in or select. "       
    )
    return seq_input


def build_tree_state(
    cleaned_html: str,
    candidate_ids: list[str],
    keep_html_brackets: bool = False,
) -> tuple[str, dict[str, int], list[list[str]]]:
    """Convert a raw Mind2Web DOM snapshot into the model-visible tree state.

    Args:
        cleaned_html: The fixed offline DOM snapshot for one Mind2Web step.
        candidate_ids: Backend node ids used to prune the DOM and define the
            action space visible to the policy.
        keep_html_brackets: Whether to preserve angle-bracket style HTML in
            Mind2Web's tree representation.

    Returns:
        tree_repr: The pruned DOM text that forms the first half of state S.
        id_mapping: Mapping from backend_node_id to local tree ids.
        choices: Candidate nodes in pruned-DOM order as
            [backend_node_id, short_node_repr].
    """

    dom_tree = lxml.etree.fromstring(cleaned_html)
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree,
        id_mapping={},
        keep_html_brackets=keep_html_brackets,
    )

    choices: list[list[str]] = []
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    for node in candidate_nodes:
        short_repr = " ".join(
            get_tree_repr(
                node,
                id_mapping=id_mapping,
                keep_html_brackets=keep_html_brackets,
            )[0].split()[:10]
        )
        choices.append([node.attrib["backend_node_id"], short_repr])

    return tree_repr, id_mapping, choices


def build_generation_target(
    sample: dict[str, Any],
    gt_backend_node_id: str | int,
    id_mapping: dict[str, int],
    choices: list[list[str]],
) -> str:
    """Build Mind2Web's generation-mode SFT target for reward metadata.

    The target is not used as teacher-forced labels in GRPO.  We keep it in
    metadata/ground_truth so a future trajectory reward can compare policy
    actions with the benchmark labels.
    """

    gt = id_mapping.get(gt_backend_node_id, -1)
    if gt == -1:
        return "None"

    current_action_op = sample["operation"]["op"]
    current_action_value = sample["operation"]["value"]
    seq_target = f"Element: {choices[gt][1]}\n"
    seq_target += f"Action: {current_action_op}\n"
    if current_action_op != "CLICK":
        seq_target += f"Value: {current_action_value}"
    return seq_target


def build_step_prompt(tree_repr: str, seq_input: str) -> str:
    """Join the two Mind2Web state components into policy prompt S."""

    return f"{tree_repr}\n{seq_input}"


def build_step_state(
    sample: dict[str, Any],
    candidate_ids: list[str],
    previous_k: int = 5,
    keep_html_brackets: bool = False,
) -> dict[str, Any]:
    """Build the complete fixed state record for one Mind2Web step.

    State S itself is `tree_repr + seq_input`.  The extra fields are retained so
    rollout logs and later reward functions can recover the action space and
    benchmark label without reparsing the DOM.
    """

    tree_repr, id_mapping, choices = build_tree_state(
        sample["cleaned_html"],
        candidate_ids,
        keep_html_brackets=keep_html_brackets,
    )
    seq_input = build_seq_input(
        confirmed_task=sample["confirmed_task"],
        previous_actions=sample.get("previous_actions", []),
        previous_k=previous_k,
    )
    pos_ids = [candidate["backend_node_id"] for candidate in sample.get("pos_candidates", [])]
    gt_backend_node_id = pos_ids[0] if pos_ids else -1
    seq_target = build_generation_target(sample, gt_backend_node_id, id_mapping, choices)

    return {
        "tree_repr": tree_repr,
        "seq_input": seq_input,
        "state_prompt": build_step_prompt(tree_repr, seq_input),
        "choices": choices,
        "id_mapping": id_mapping,
        "candidate_ids": candidate_ids,
        "seq_target": seq_target,
    }

