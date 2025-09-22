"""Gate telemetry computations (precision/recall, PR-AUC, calibration)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["compute_gate_metrics"]


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_float(value: Any) -> float:
    return float(value)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _pr_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = sum(1 for label in labels if label)
    if positives == 0 or not scores:
        return 0.0
    pairs = sorted(zip(scores, labels, strict=False), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    last_recall = 0.0
    area = 0.0
    for _score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / positives
        if recall < last_recall:  # pragma: no cover - defensive guard
            recall = last_recall
        area += precision * max(0.0, recall - last_recall)
        last_recall = recall
    return float(area)


def _calibration(
    scores: Sequence[float],
    labels: Sequence[bool],
    bins: int,
) -> list[dict[str, float | int]]:
    if bins <= 0:
        raise ValueError("calibration bins must be positive")
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        positives = sum(1 for label in labels if label)
        count = len(scores)
        fraction = positives / count if count else 0.0
        mean = sum(scores) / count if count else 0.0
        return [
            {
                "lower": float(min_score),
                "upper": float(max_score),
                "count": count,
                "positives": positives,
                "fraction_positive": fraction,
                "mean_score": float(mean),
            }
        ]
    width = (max_score - min_score) / bins
    buckets: list[dict[str, float | int]] = [
        {
            "lower": float(min_score + i * width),
            "upper": float(min_score + (i + 1) * width),
            "count": 0,
            "positives": 0,
            "fraction_positive": 0.0,
            "mean_score": 0.0,
            "_score_sum": 0.0,
        }
        for i in range(bins)
    ]
    for score, label in zip(scores, labels, strict=False):
        offset = score - min_score
        idx = int(offset / width) if width else 0
        if idx >= bins:
            idx = bins - 1
        bucket = buckets[idx]
        bucket["count"] = int(bucket["count"]) + 1
        bucket["positives"] = int(bucket["positives"]) + int(label)
        bucket["_score_sum"] = float(bucket["_score_sum"]) + float(score)
    for bucket in buckets:
        count = int(bucket["count"])
        score_sum = float(bucket.pop("_score_sum"))
        bucket["fraction_positive"] = (int(bucket["positives"]) / count) if count else 0.0
        bucket["mean_score"] = (score_sum / count) if count else 0.0
    return buckets


def compute_gate_metrics(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    calibration_bins: int = 10,
) -> dict[str, float | int | None | list[dict[str, float | int]]]:
    """Return aggregate telemetry metrics for gate diagnostics."""

    total = len(diagnostics)
    if total == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "pr_auc": 0.0,
            "clutter_rate": 0.0,
            "writes": 0,
            "positives": 0,
            "total": 0,
            "calibration": [],
            "writes_per_1k_records": 0.0,
            "writes_per_1k_tokens": None,
            "generated_tokens": 0,
            "prompt_tokens": 0,
        }
    writes = sum(1 for diag in diagnostics if _safe_bool(diag.get("should_write")))
    positives = sum(1 for diag in diagnostics if _safe_bool(diag.get("should_remember_label")))
    tp = sum(
        1
        for diag in diagnostics
        if _safe_bool(diag.get("should_write")) and _safe_bool(diag.get("should_remember_label"))
    )
    fp = writes - tp
    fn = positives - tp
    precision, recall = _precision_recall(tp, fp, fn)
    scores = [_safe_float(diag.get("score", 0.0)) for diag in diagnostics]
    labels = [_safe_bool(diag.get("should_remember_label")) for diag in diagnostics]
    pr_auc = _pr_auc(scores, labels)
    calibration = _calibration(scores, labels, calibration_bins)
    clutter_rate = writes / total if total else 0.0
    generated_tokens = sum(_safe_int(diag.get("generated_tokens")) for diag in diagnostics)
    prompt_tokens = sum(_safe_int(diag.get("prompt_tokens")) for diag in diagnostics)
    if generated_tokens > 0:
        writes_per_1k_tokens = writes / (generated_tokens / 1000.0)
    else:
        writes_per_1k_tokens = None
    return {
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "clutter_rate": clutter_rate,
        "writes": writes,
        "positives": positives,
        "total": total,
        "calibration": calibration,
        "writes_per_1k_records": clutter_rate * 1000.0,
        "writes_per_1k_tokens": writes_per_1k_tokens,
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
    }
