"""Gate telemetry computations (precision/recall, PR-AUC, calibration)."""

from __future__ import annotations

import math
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


def _empty_score_distribution() -> dict[str, Any]:
    return {
        "p10": None,
        "p50": None,
        "p90": None,
        "histogram": [],
    }


def _percentile(sorted_scores: Sequence[float], fraction: float) -> float:
    if not sorted_scores:
        return 0.0
    if fraction <= 0.0:
        return float(sorted_scores[0])
    if fraction >= 1.0:
        return float(sorted_scores[-1])
    position = fraction * (len(sorted_scores) - 1)
    lower_idx = math.floor(position)
    upper_idx = math.ceil(position)
    if lower_idx == upper_idx:
        return float(sorted_scores[int(position)])
    lower = float(sorted_scores[lower_idx])
    upper = float(sorted_scores[upper_idx])
    weight = position - lower_idx
    return lower + weight * (upper - lower)


def _score_distribution(scores: Sequence[float], bins: int) -> dict[str, Any]:
    if bins <= 0:
        raise ValueError("score histogram bins must be positive")
    if not scores:
        return _empty_score_distribution()
    sorted_scores = sorted(float(score) for score in scores)
    p10 = _percentile(sorted_scores, 0.10)
    p50 = _percentile(sorted_scores, 0.50)
    p90 = _percentile(sorted_scores, 0.90)
    min_score = sorted_scores[0]
    max_score = sorted_scores[-1]
    histogram: list[dict[str, float | int]]
    if min_score == max_score:
        histogram = [
            {
                "lower": float(min_score),
                "upper": float(max_score),
                "count": int(len(sorted_scores)),
            }
        ]
    else:
        width = (max_score - min_score) / bins
        histogram = []
        counts = [0 for _ in range(bins)]
        for score in sorted_scores:
            if width == 0:
                idx = 0
            else:
                idx = int((score - min_score) / width)
            if idx >= bins:
                idx = bins - 1
            counts[idx] += 1
        for idx, count in enumerate(counts):
            lower = min_score + idx * width
            upper = min_score + (idx + 1) * width
            if idx == bins - 1:
                upper = max_score
            histogram.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(count),
                }
            )
    return {
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "histogram": histogram,
    }


def _precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _pr_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = sum(1 for label in labels if label)
    total = len(labels)
    if positives == 0 or not scores:
        return 0.0
    if positives == total:
        return 1.0
    grouped: dict[float, list[bool]] = {}
    for score, label in zip(scores, labels, strict=False):
        key = float(score)
        grouped.setdefault(key, []).append(bool(label))
    sorted_scores = sorted(grouped, reverse=True)
    tp = 0
    fp = 0
    last_recall = 0.0
    area = 0.0
    for key in sorted_scores:
        labels_for_score = grouped[key]
        positive_in_group = sum(1 for label in labels_for_score if label)
        negative_in_group = len(labels_for_score) - positive_in_group
        tp += positive_in_group
        fp += negative_in_group
        precision = tp / (tp + fp) if (tp + fp) else 0.0
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
) -> dict[str, Any]:
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
            "score_distribution": _empty_score_distribution(),
            "writes_per_1k_records": 0.0,
            "writes_per_1k_tokens": None,
            "generated_tokens": 0,
            "prompt_tokens": 0,
            "label_distribution": {
                "positives": 0,
                "negatives": 0,
                "positive_rate": 0.0,
            },
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
    score_distribution = _score_distribution(scores, calibration_bins)
    clutter_rate = writes / total if total else 0.0
    generated_tokens = sum(_safe_int(diag.get("generated_tokens")) for diag in diagnostics)
    prompt_tokens = sum(_safe_int(diag.get("prompt_tokens")) for diag in diagnostics)
    total_tokens = generated_tokens + prompt_tokens
    if total_tokens > 0:
        writes_per_1k_tokens = writes / (total_tokens / 1000.0)
    else:
        writes_per_1k_tokens = None
    negatives = max(total - positives, 0)
    positive_rate = positives / total if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "clutter_rate": clutter_rate,
        "writes": writes,
        "positives": positives,
        "total": total,
        "calibration": calibration,
        "score_distribution": score_distribution,
        "writes_per_1k_records": clutter_rate * 1000.0,
        "writes_per_1k_tokens": writes_per_1k_tokens,
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "label_distribution": {
            "positives": positives,
            "negatives": negatives,
            "positive_rate": positive_rate,
        },
    }
