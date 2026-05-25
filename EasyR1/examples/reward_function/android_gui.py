"""
Number Game Dense Reward Function

四个维度（加权求和到 overall）：
    1. format —— 输出是否符合 <think>...</think><answer>X</answer> 结构
    2. reason —— 推理过程是否合理、自洽（image-grounded LLM-as-Judge）
    3. budget —— 推理 token 数是否在预算范围内
    4. final  —— 最终 <answer> 是否等于 ground_truth

Judge 设计（image-grounded、group-wise 比较）：
- Judge 看到的是 IMAGE + 同 group 内全部 N 条 rollout 的 trace；**不提供 ground truth**。
- Judge 自己用 VLM 能力从 image 中恢复 (Light, Numbers)，再独立比对每条 trace 的声明
  是否如实描述了图像；并基于这一独立判断给出 6 个 axis 的分数。
- 同时强制输出严格 ranking（无并列）→ 派生 rank_score，组内永远有 σ>0。
- 同 group 共享一次 API 调用，节省成本；image base64 在日志里会被替换成占位符。

判分轴（每个 ∈ [0,1]）：
    light_correct, numbers_correct, rule_correct, choice_follows, clarity, conciseness
reason = 0.6 * mean(axes) + 0.4 * rank_score

可配置项（通过 reward_function_kwargs 透传）：
    weights:                {format, reason, budget, final}
    budget_target_low/high/hard_max: int

环境变量（由 verl/trainer/main.py 转发到 Ray worker）：
    JUDGE_ENABLED         "true"/"false" (默认 false)
    JUDGE_API_KEY         OpenAI / 兼容 API 的 key
    JUDGE_BASE_URL        默认 https://api.openai.com/v1
    JUDGE_MODEL           需要是 VLM；默认 gpt-4o-mini (vision capable)
    JUDGE_TIMEOUT_S       默认 30
    JUDGE_MAX_WORKERS     默认 8
    JUDGE_TEMPERATURE     默认 0

依赖：
    pip install openai pillow            # 仅在 JUDGE_ENABLED=true 时需要
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Metadata —— EasyR1 框架要求
REWARD_NAME = "number_game"
REWARD_TYPE = "batch"


# ============================================================
# 1. 解析 utilities
# ============================================================

_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*([012])\s*</answer>", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_FIELD_RE: Dict[str, re.Pattern] = {
    "light": re.compile(r"Light:\s*(GREEN|RED|YELLOW)", re.IGNORECASE),
    "numbers": re.compile(
        r"Numbers:\s*left\s*=\s*(-?\d+),\s*middle\s*=\s*(-?\d+),\s*right\s*=\s*(-?\d+)",
        re.IGNORECASE,
    ),
    "rule": re.compile(r"Rule:\s*(.+)", re.IGNORECASE),
    "choice": re.compile(r"Choice:\s*position\s*([012])", re.IGNORECASE),
}


def extract_answer(response: str) -> str:
    """优先匹配 <answer>X</answer>，再退回旧逻辑。"""
    match = _ANSWER_BLOCK_RE.search(response)
    if match:
        return match.group(1)
    stripped = response.strip()
    if stripped in {"0", "1", "2"}:
        return stripped
    match = re.search(r"[012]", response)
    return match.group(0) if match else ""


def parse_think(response: str) -> Dict[str, Any]:
    """从 <think> 中抽取结构化字段；缺失字段为 None。"""
    fields: Dict[str, Any] = {"light": None, "numbers": None, "rule": None, "choice": None}
    block = _THINK_BLOCK_RE.search(response)
    if not block:
        return fields
    text = block.group(1)
    for key, pattern in _THINK_FIELD_RE.items():
        match = pattern.search(text)
        if not match:
            continue
        if key == "numbers":
            fields[key] = [int(match.group(i)) for i in (1, 2, 3)]
        elif key == "light":
            fields[key] = match.group(1).strip().upper()
        else:
            fields[key] = match.group(1).strip()
    return fields


# ============================================================
# 2. 子项打分
# ============================================================

def score_format(response: str) -> Dict[str, float]:
    """格式分（连续）：
    - think 块存在            → 0.20
    - answer 块存在 (合法 0/1/2) → 0.30
    - <think> 在 <answer> 之前  → 0.10
    - 4 个 think 字段每个独立加  → 0.10 * 字段权重 = 0.40

    字段权重：Light/Numbers/Choice 各 0.30，Rule 0.10（Rule 是自由文本，最容易写但最不重要）。
    所有子项都是 [0,1]，主分 ∈ [0,1]。
    """
    has_think = _THINK_BLOCK_RE.search(response) is not None
    answer_match = _ANSWER_BLOCK_RE.search(response)
    has_answer = answer_match is not None

    think_match = _THINK_BLOCK_RE.search(response)
    order_ok = bool(
        has_think and has_answer and think_match.start() < answer_match.start()
    )

    fields = parse_think(response)
    field_weights = {"light": 0.30, "numbers": 0.30, "choice": 0.30, "rule": 0.10}
    fields_score = sum(w for k, w in field_weights.items() if fields[k] is not None)

    overall = (
        0.20 * float(has_think)
        + 0.30 * float(has_answer)
        + 0.10 * float(order_ok)
        + 0.40 * fields_score
    )
    return {
        "overall": overall,
        "has_think": float(has_think),
        "has_answer": float(has_answer),
        "order_ok": float(order_ok),
        "fields_complete": fields_score,
    }


def score_budget(response_length: int, target_low: int, target_high: int, hard_max: int) -> float:
    """长度预算分（连续、单调、无平台）：
    用一个连续三角/线性函数，让任意两个不同长度都得不同分。
    - length <= 0                  → 0.0
    - 0 < length <= target_low     → 线性 0 → 1（鼓励有最低限度的推理长度）
    - target_low < length <= target_high → 线性 1 → 0.5
    - target_high < length <= hard_max   → 线性 0.5 → 0
    - length > hard_max            → 0.0

    与旧版的关键差异：
    - 不再有"<= low 全部 1.0"的平台，避免所有合理长度打平
    - 整段单调，length 每差 1 token 都体现在分数上
    """
    if response_length <= 0:
        return 0.0
    if response_length <= target_low:
        # 从 0 升到 1，避免极短输出反而拿满分
        return response_length / max(target_low, 1)
    if response_length <= target_high:
        ratio = (response_length - target_low) / max(target_high - target_low, 1)
        return 1.0 - 0.5 * ratio
    if response_length <= hard_max:
        ratio = (response_length - target_high) / max(hard_max - target_high, 1)
        return max(0.0, 0.5 - 0.5 * ratio)
    return 0.0


def score_final(response: str, ground_truth: str) -> float:
    return 1.0 if extract_answer(response) == ground_truth else 0.0


# ============================================================
# 3. LLM-as-Judge
# ============================================================

_JUDGE_SYSTEM_PROMPT = """You are an expert grader for a visual reasoning task. You will see:
1. ONE IMAGE showing a screen with a traffic light and three numbers at positions left=0, middle=1, right=2.
2. N CANDIDATE reasoning traces from different model rollouts. Each trace contains:
   - A `<think>` block listing Light / Numbers / Rule / Choice.
   - A final `<answer>X</answer>` with X in {0, 1, 2}.

Game rule:
- GREEN light -> the correct answer is the position of the LARGEST number.
- RED   light -> the position of the SMALLEST number.
- YELLOW light -> the position of the MIDDLE number.

IMPORTANT: You are NOT given the ground-truth answer. You must judge each trace INDEPENDENTLY by:
(a) inspecting the image yourself to recover the true light color and the three numbers,
(b) checking each trace's claims against what the image actually shows,
(c) checking that the trace's chosen rule and final position follow from those claims.

CRITICAL output rules:
1. "ranking" MUST be a strict permutation of [0..N-1] with NO TIES. First = best, last = worst.
2. Use the full [0, 1] range; AVOID clustering at 1.0 or 0.5. Even if every trace is fully correct, differentiate by clarity and conciseness.
3. Be strict about correctness against the image; lenient about wording.
4. If a trace has no reasoning at all, give 0 across the board for that trace.

Axes per candidate (each float in [0, 1]):
- light_correct:    the trace's stated Light matches the light actually shown in the image.
- numbers_correct:  the trace's stated Numbers (left/middle/right) match the image.
- rule_correct:     the chosen Rule matches the stated Light per game rules.
- choice_follows:   the final Choice follows logically from (stated Numbers, stated Rule).
- clarity:          reasoning is clearly written and easy to follow.
- conciseness:      not redundant, not unnecessarily verbose.

Return ONLY a compact JSON object with this EXACT schema (no prose, no markdown):
{
  "image_observation": {"light": "GREEN|RED|YELLOW", "numbers": [left, middle, right]},
  "ratings": [
    {"index": 0, "light_correct": float, "numbers_correct": float, "rule_correct": float, "choice_follows": float, "clarity": float, "conciseness": float},
    {"index": 1, ...},
    ...
  ],
  "ranking": [best_idx, ..., worst_idx],   // strict permutation of [0..N-1], NO TIES
  "rationale": "one short sentence explaining the ranking"
}
"""

# axis defaults + auxiliary fields used downstream
_NEUTRAL_JUDGE: Dict[str, Any] = {
    "light_correct": 0.5,
    "numbers_correct": 0.5,
    "rule_correct": 0.5,
    "choice_follows": 0.5,
    "clarity": 0.5,
    "conciseness": 0.5,
    "rank_score": 0.5,        # 由 ranking 派生：1 - rank/(N-1)
    "ranking_pos": None,      # 该 rollout 在 ranking 中的位置（0=best）
    "verdict": "ok",
}
_JUDGE_AXES = ("light_correct", "numbers_correct", "rule_correct", "choice_follows", "clarity", "conciseness")


class JudgeConfig:
    """所有 judge 配置都从环境变量读取。"""

    def __init__(self) -> None:
        self.enabled: bool = os.getenv("JUDGE_ENABLED", "false").strip().lower() == "true"
        self.api_key: Optional[str] = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url: str = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
        self.model: str = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
        self.timeout: float = float(os.getenv("JUDGE_TIMEOUT_S", "30"))
        self.max_workers: int = int(os.getenv("JUDGE_MAX_WORKERS", "8"))
        self.temperature: float = float(os.getenv("JUDGE_TEMPERATURE", "0"))

    def is_ready(self) -> bool:
        return self.enabled and bool(self.api_key)


def _encode_image_to_data_url(image: Any) -> Optional[str]:
    """把 PIL.Image / 文件路径 / URL 转换为 OpenAI 多模态 API 接受的字符串。
    PIL Image 或本地路径 → data URL (base64)；http URL 直接返回。失败返回 None。
    """
    try:
        if image is None:
            return None
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return image
            if os.path.exists(image):
                with open(image, "rb") as f:
                    raw = f.read()
                mime = "image/jpeg" if image.lower().endswith((".jpg", ".jpeg")) else "image/png"
                return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
            return None
        # 假设是 PIL Image
        if hasattr(image, "save"):
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("encode image failed: %s", exc)
    return None


def _build_group_judge_user_content(responses: List[str], image_urls: List[str]) -> List[Dict[str, Any]]:
    """多模态 user content：先把所有 image 贴上，再贴 N 条 rollout 的文本。"""
    content: List[Dict[str, Any]] = []
    if image_urls:
        content.append({
            "type": "text",
            "text": (
                f"You are looking at the screen the model saw. "
                f"Below is/are {len(image_urls)} image(s); inspect them to recover the true Light and Numbers."
            ),
        })
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        content.append({
            "type": "text",
            "text": (
                "(No image was available; judge based only on the trace's internal consistency. "
                "Mark light_correct and numbers_correct as 0.5 if you cannot verify.)"
            ),
        })

    text_parts: List[str] = [
        "",
        f"Below are {len(responses)} candidate reasoning traces from different model rollouts for the SAME image:",
        "",
    ]
    for i, r in enumerate(responses):
        text_parts.append(f"--- Candidate {i} (verbatim) ---")
        text_parts.append("```")
        text_parts.append(r)
        text_parts.append("```")
        text_parts.append("")
    text_parts.append(
        "Grade ALL candidates against the image and produce a STRICT ranking with NO TIES. "
        "Output JSON only with the exact schema described in the system prompt."
    )
    content.append({"type": "text", "text": "\n".join(text_parts)})
    return content


def _redact_image_payload(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把消息里 image_url 的 base64 payload 替换成占位符，避免日志 JSON 爆炸。"""
    redacted: List[Dict[str, Any]] = []
    for msg in messages:
        new_msg = {"role": msg.get("role")}
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        size = len(url)
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"<base64 image, {size} chars, redacted>"},
                        })
                    else:
                        new_content.append(part)
                else:
                    new_content.append(part)
            new_msg["content"] = new_content
        else:
            new_msg["content"] = content
        redacted.append(new_msg)
    return redacted


def _safe_rating(raw: Any) -> Dict[str, float]:
    """把一条 rating dict 规整到 [0,1] 区间，缺失字段填 0.5。"""
    out = {axis: 0.5 for axis in _JUDGE_AXES}
    if not isinstance(raw, dict):
        return out
    for axis in _JUDGE_AXES:
        try:
            out[axis] = max(0.0, min(1.0, float(raw.get(axis, 0.5))))
        except (TypeError, ValueError):
            pass
    return out


def _safe_ranking(raw: Any, n: int) -> List[int]:
    """校验 ranking：必须是 [0..n-1] 的严格置换；否则退化为顺序排列。"""
    if not isinstance(raw, list) or len(raw) != n:
        return list(range(n))
    try:
        ints = [int(x) for x in raw]
    except (TypeError, ValueError):
        return list(range(n))
    if sorted(ints) != list(range(n)):
        return list(range(n))
    return ints


def _make_judge_log(
    enabled: bool,
    model: str,
    parsed: Dict[str, Any],
    messages: Optional[List[Dict[str, str]]] = None,
    response_raw: Optional[str] = None,
    error: Optional[str] = None,
    group_uid: Optional[str] = None,
    group_size: Optional[int] = None,
) -> Dict[str, Any]:
    """统一格式的 judge 轨迹日志，写进 rollout JSON 里。
    parsed 包含：6 个 axis + rank_score + ranking_pos + verdict
    """
    return {
        "enabled": enabled,
        "model": model,
        "group_uid": group_uid,
        "group_size": group_size,
        "messages": messages,           # judge 实际发送的 prompt（system + user）；group 内 N 个 rollout 共享
        "response_raw": response_raw,   # judge 返回的原始字符串；group 内共享
        "parsed": parsed,               # 该 rollout 自己的 axis 分数 + rank_score
        "error": error,                 # 如果失败，记录异常字符串
    }


def _neutral_logs(
    n: int, model: str, group_uid: str, error: str
) -> List[Dict[str, Any]]:
    """构造一个 group 的 N 条中性 fallback 日志（judge 未开 / 失败时用）。"""
    return [
        _make_judge_log(
            enabled=False if "disabled" in error else True,
            model=model,
            parsed=dict(_NEUTRAL_JUDGE),
            error=error,
            group_uid=group_uid,
            group_size=n,
        )
        for _ in range(n)
    ]


def judge_one_group(
    client, model: str, group_responses: List[str], group_images: Optional[List[Any]],
    temperature: float, timeout: float, group_uid: str,
) -> List[Dict[str, Any]]:
    """对一个 group 内的 N 条 rollout 做一次 image-grounded 比较式打分。
    judge 看到的是 image + N 条 trace（**不含 ground truth**），自己用 VLM 能力判断每条 trace 是否如实描述了图像。"""
    n = len(group_responses)

    # 把 image 编码成 data URL（只取第一张；同 group 共享）
    image_urls: List[str] = []
    if group_images:
        for img in group_images[:1]:  # 一个 prompt 一般只有一张图；多图可改这里
            url = _encode_image_to_data_url(img)
            if url:
                image_urls.append(url)

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _build_group_judge_user_content(group_responses, image_urls)},
    ]
    redacted_messages = _redact_image_payload(messages)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    raw_text = completion.choices[0].message.content
    try:
        raw_obj = json.loads(raw_text)
    except (TypeError, ValueError) as exc:
        return [
            _make_judge_log(
                True, model, dict(_NEUTRAL_JUDGE),
                messages=redacted_messages, response_raw=raw_text,
                error=f"json_parse_error: {exc!r}",
                group_uid=group_uid, group_size=n,
            )
            for _ in range(n)
        ]

    ratings_raw = raw_obj.get("ratings", [])
    ranking = _safe_ranking(raw_obj.get("ranking"), n)
    image_observation = raw_obj.get("image_observation")
    rationale = raw_obj.get("rationale")

    # rank → rank_score: best=1.0, worst=0.0
    rank_score_by_idx: Dict[int, float] = {}
    rank_pos_by_idx: Dict[int, int] = {}
    for rank, idx in enumerate(ranking):
        rank_score_by_idx[idx] = 1.0 - rank / max(n - 1, 1)
        rank_pos_by_idx[idx] = rank

    # 按 candidate index 查 rating；查不到就用 list 位置
    rating_by_idx: Dict[int, Dict[str, Any]] = {}
    for pos, r in enumerate(ratings_raw if isinstance(ratings_raw, list) else []):
        if isinstance(r, dict) and isinstance(r.get("index"), int):
            rating_by_idx[r["index"]] = r
        else:
            rating_by_idx[pos] = r if isinstance(r, dict) else {}

    logs: List[Dict[str, Any]] = []
    for local_idx in range(n):
        parsed = _safe_rating(rating_by_idx.get(local_idx, {}))
        parsed["rank_score"] = rank_score_by_idx.get(local_idx, 0.5)
        parsed["ranking_pos"] = rank_pos_by_idx.get(local_idx)
        if isinstance(rationale, str):
            parsed["verdict"] = rationale[:200]
        if isinstance(image_observation, dict):
            parsed["image_observation"] = image_observation
        logs.append(_make_judge_log(
            True, model, parsed,
            messages=redacted_messages, response_raw=raw_text, error=None,
            group_uid=group_uid, group_size=n,
        ))
    return logs


def judge_grouped(
    reward_inputs: List[Dict[str, Any]], config: JudgeConfig
) -> List[Dict[str, Any]]:
    """按 uid 分组，每组一次 (多模态) API 调用。返回与输入等长的 judge_log 列表。"""
    n = len(reward_inputs)
    logs: List[Optional[Dict[str, Any]]] = [None] * n

    # 按 uid 分组
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, ri in enumerate(reward_inputs):
        groups[str(ri.get("uid", f"sample_{i}"))].append(i)

    def _fill_neutral(group_uid: str, indices: List[int], error: str) -> None:
        fallback = _neutral_logs(len(indices), config.model, group_uid, error)
        for local, global_i in enumerate(indices):
            logs[global_i] = fallback[local]

    if not config.is_ready():
        for uid, indices in groups.items():
            _fill_neutral(uid, indices, "judge_disabled_or_no_api_key")
        return [log for log in logs if log is not None]

    try:
        from openai import OpenAI  # lazy import
    except ImportError as exc:
        logger.warning("openai package not installed; judge disabled, using neutral scores.")
        for uid, indices in groups.items():
            _fill_neutral(uid, indices, f"openai_import_error: {exc}")
        return [log for log in logs if log is not None]

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def _task(item: Tuple[str, List[int]]) -> None:
        uid, indices = item
        group_responses = [reward_inputs[i]["response"] for i in indices]
        group_images = reward_inputs[indices[0]].get("images")  # 同 group 共享 image
        try:
            group_logs = judge_one_group(
                client, config.model, group_responses, group_images,
                config.temperature, config.timeout, uid,
            )
            for local, global_i in enumerate(indices):
                logs[global_i] = group_logs[local]
        except Exception as exc:  # noqa: BLE001
            logger.warning("group judge call failed (uid=%s, model=%s): %s; using neutral.",
                           uid, config.model, exc)
            _fill_neutral(uid, indices, f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as pool:
        list(pool.map(_task, list(groups.items())))

    # logs 应已全部填充；保险起见把残余 None 替换成中性
    return [
        log if log is not None
        else _make_judge_log(False, config.model, dict(_NEUTRAL_JUDGE),
                             error="missing_log_unexpected")
        for log in logs
    ]


def score_reason_from_judge(judge_out: Dict[str, Any]) -> float:
    """聚合 judge 多维分数到 [0,1]。
    设计：axes 提供绝对质量信号，rank_score 提供组内相对排序信号（保证组内永远有 σ>0）。
    reason = 0.6 * axes_mean + 0.4 * rank_score
    """
    axes_values = [float(judge_out.get(axis, 0.5)) for axis in _JUDGE_AXES]
    axes_mean = sum(axes_values) / len(axes_values)
    rank_score = float(judge_out.get("rank_score", axes_mean))
    return max(0.0, min(1.0, 0.6 * axes_mean + 0.4 * rank_score))


# ============================================================
# 4. 聚合：compute_score
# ============================================================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "format": 0.15,
    "reason": 0.25,
    "budget": 0.10,
    "final": 0.50,
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    budget_target_low: int = 80,
    budget_target_high: int = 180,
    budget_hard_max: int = 256,
) -> List[Dict[str, float]]:
    """EasyR1 batch reward function.

    Args:
        reward_inputs: list of {"response", "response_length", "ground_truth"}.
        weights: 可选覆盖 DEFAULT_WEIGHTS。
        budget_target_low/high/hard_max: 长度预算阈值。

    Returns:
        与输入等长的列表，每项包含 overall + 各子项 + judge 多维度（仅监控）。
    """
    final_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    judge_config = JudgeConfig()

    responses = [str(r.get("response", "")) for r in reward_inputs]
    ground_truths = [str(r.get("ground_truth", "")) for r in reward_inputs]
    lengths = [int(r.get("response_length", 0)) for r in reward_inputs]

    judge_logs = judge_grouped(reward_inputs, judge_config)

    out: List[Dict[str, Any]] = []
    for response, gt, length, judge_log in zip(responses, ground_truths, lengths, judge_logs):
        fmt = score_format(response)
        bud = score_budget(length, budget_target_low, budget_target_high, budget_hard_max)
        fin = score_final(response, gt)
        parsed = judge_log["parsed"]
        rea = score_reason_from_judge(parsed)

        overall = (
            final_weights["format"] * fmt["overall"]
            + final_weights["reason"] * rea
            + final_weights["budget"] * bud
            + final_weights["final"] * fin
        )

        out.append({
            "overall": float(overall),
            # 主轴
            "format": fmt["overall"],
            "reason": rea,
            "budget": bud,
            "accuracy": fin,
            # 监控子项
            "format_has_think": fmt["has_think"],
            "format_has_answer": fmt["has_answer"],
            "format_order_ok": fmt["order_ok"],
            "format_fields_complete": fmt["fields_complete"],
            "judge_light_correct": float(parsed.get("light_correct", 0.5)),
            "judge_numbers_correct": float(parsed.get("numbers_correct", 0.5)),
            "judge_rule_correct": float(parsed.get("rule_correct", 0.5)),
            "judge_choice_follows": float(parsed.get("choice_follows", 0.5)),
            "judge_clarity": float(parsed.get("clarity", 0.5)),
            "judge_conciseness": float(parsed.get("conciseness", 0.5)),
            "judge_rank_score": float(parsed.get("rank_score", 0.5)),
            # 非数值 trace（约定 `_` 前缀的 key 由 reward manager 拆到 extras，不进 metrics 均值）
            "_judge_log": judge_log,
        })
    return out


# ============================================================
# 5. Self-test（judge 关闭时用 fallback；不依赖网络）
# ============================================================

if __name__ == "__main__":
    # 模拟一个 group：同一 uid 的 3 个 rollout（与 step_0001.json 中的情况一致）
    group_a_uid = "group-a"
    # 第二个 group：1 个 rollout，演示完全不同的样本
    group_b_uid = "group-b"
    cases = [
        {
            "response": (
                "<think>\nLight: GREEN\nNumbers: left=2, middle=7, right=5\n"
                "Rule: Select the largest number when the light is green.\n"
                "Choice: position 1 because the largest number is 7.\n</think>\n<answer>1</answer>"
            ),
            "response_length": 57, "ground_truth": "1", "uid": group_a_uid,
        },
        {
            "response": (
                "<think>\nLight: GREEN\nNumbers: left=2, middle=7, right=5\n"
                "Rule: Select the largest number for GREEN light\n"
                "Choice: position 1 because the largest number is 7\n</think>\n<answer>1</answer>"
            ),
            "response_length": 55, "ground_truth": "1", "uid": group_a_uid,
        },
        {
            "response": (
                "<think>\nLight: GREEN\nNumbers: left=2, middle=7, right=5\n"
                "Rule: Select the largest number\n"
                "Choice: position 1 because 7 is the largest number\n</think>\n<answer>1</answer>"
            ),
            "response_length": 52, "ground_truth": "1", "uid": group_a_uid,
        },
        {
            "response": "1",
            "response_length": 1, "ground_truth": "1", "uid": group_b_uid,
        },
    ]

    scored = compute_score(cases)
    for i, sc in enumerate(scored, 1):
        print(f"--- Case {i} (uid={cases[i-1]['uid']}) ---")
        print(json.dumps(sc, indent=2, ensure_ascii=False))
