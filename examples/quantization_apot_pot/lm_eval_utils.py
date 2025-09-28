#!/usr/bin/env python3
"""Shared helpers for running `lm_eval` from quantization examples."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Keywords that usually indicate chat/instruction-tuned checkpoints
_CHAT_KEYWORDS = ("instruct", "chat", "-it", "-sft", "-align")


def parse_extra_model_args(extra: Optional[Sequence[str]]) -> List[str]:
    """Normalize user-provided key=value pairs for `lm_eval` model args."""

    if extra is None:
        return []

    if isinstance(extra, str):
        extra_iter: Iterable[str] = extra.split(",")
    else:
        extra_iter = extra

    parsed: List[str] = []
    for item in extra_iter:
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected key=value pair for extra model arg, got: {token}")
        parsed.append(token)
    return parsed


def should_auto_apply_chat_template(pretrained: str) -> bool:
    """Heuristic to decide whether to turn on chat templates automatically."""

    lowered = pretrained.lower()
    return any(keyword in lowered for keyword in _CHAT_KEYWORDS)


def build_model_args(
    pretrained: str,
    *,
    device: Optional[str] = None,
    use_accelerate: bool = False,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    extra: Optional[Sequence[str]] = None,
) -> str:
    """Construct the comma-separated `model_args` string for `lm_eval`."""

    parts: List[str] = [f"pretrained={pretrained}"]

    if device:
        parts.append(f"device={device}")
    if use_accelerate:
        parts.append("use_accelerate=True")
    if apply_chat_template:
        parts.append("apply_chat_template=True")
        if fewshot_as_multiturn:
            parts.append("fewshot_as_multiturn=True")

    for kv in parse_extra_model_args(extra):
        parts.append(kv)

    return ",".join(parts)


def run_lm_eval(
    *,
    pretrained: str,
    tasks: Sequence[str] | str,
    model_kind: str = "hf",
    batch_size: str | int = "2",
    num_fewshot: int = 0,
    limit: Optional[str] = None,
    device: Optional[str] = None,
    use_accelerate: bool = False,
    apply_chat_template: Optional[bool] = None,
    fewshot_as_multiturn: bool = False,
    extra_model_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute `lm_eval`'s simple_evaluate helper with shared defaults."""

    try:
        from lm_eval import evaluator  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency import guard
        raise ImportError(
            "lm-eval is not installed. Install with: pip install -U lm-eval"
        ) from exc

    task_list = (
        [t.strip() for t in tasks.split(",") if t.strip()]
        if isinstance(tasks, str)
        else [t.strip() for t in tasks if t and t.strip()]
    )

    if not task_list:
        raise ValueError("At least one evaluation task must be provided")

    auto_chat = should_auto_apply_chat_template(pretrained)
    apply_chat = auto_chat if apply_chat_template is None else apply_chat_template

    model_args = build_model_args(
        pretrained=pretrained,
        device=device,
        use_accelerate=use_accelerate,
        apply_chat_template=apply_chat,
        fewshot_as_multiturn=fewshot_as_multiturn,
        extra=extra_model_args,
    )

    return evaluator.simple_evaluate(
        model=model_kind,
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )


def collect_perplexity_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract metrics ending with `perplexity` from an lm_eval result payload."""

    metrics: Dict[str, Dict[str, float]] = {}
    if not results:
        return metrics

    per_task = results.get("results", {})
    for task, task_metrics in per_task.items():
        if not isinstance(task_metrics, dict):
            continue
        for metric_name, value in task_metrics.items():
            if "perplexity" not in metric_name.lower():
                continue
            if isinstance(value, (int, float)):
                metrics.setdefault(task, {})[metric_name] = float(value)
    return metrics


def compute_perplexity_delta(
    baseline: Dict[str, Dict[str, float]],
    candidate: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Return candidate-baseline difference for overlapping perplexity metrics."""

    delta: Dict[str, Dict[str, float]] = {}
    for task, metric_map in candidate.items():
        for metric_name, value in metric_map.items():
            ref = baseline.get(task, {}).get(metric_name)
            if ref is None:
                continue
            delta.setdefault(task, {})[metric_name] = value - ref
    return delta


def summarize_perplexity(label: str, metrics: Dict[str, Dict[str, float]]) -> str:
    """Create a human-friendly summary string for perplexity metrics."""

    lines = [label]
    if not metrics:
        lines.append("  (no perplexity metrics reported)")
        return "\n".join(lines)

    for task in sorted(metrics):
        pieces = [f"{name}={value:.4f}" for name, value in sorted(metrics[task].items())]
        lines.append(f"  - {task}: {', '.join(pieces)}")
    return "\n".join(lines)


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    """Persist evaluation payload to disk."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


