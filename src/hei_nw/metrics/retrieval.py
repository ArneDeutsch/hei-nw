"""Retrieval quality metrics for Scenario A experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def precision_at_k(
    candidates: Sequence[Sequence[int]],
    truths: Sequence[int],
    k: int,
) -> float:
    """Compute precision@k over retrieval results.

    Parameters
    ----------
    candidates:
        For each query a sequence of retrieved ``group_id`` values ordered by
        decreasing score.
    truths:
        True ``group_id`` for each query.
    k:
        Number of top candidates to consider.

    Returns
    -------
    float
        Mean precision@k across queries. Returns ``0.0`` when ``k`` is non-positive
        or inputs are empty.
    """

    if k <= 0 or not candidates or not truths:
        return 0.0
    hits = 0
    total = 0
    for cand, truth in zip(candidates, truths, strict=False):
        if truth in cand[:k]:
            hits += 1
        total += 1
    return hits / total if total else 0.0


def mrr(candidates: Sequence[Sequence[int]], truths: Sequence[int]) -> float:
    """Compute mean reciprocal rank for retrieval results."""

    if not candidates or not truths:
        return 0.0
    rr_sum = 0.0
    total = 0
    for cand, truth in zip(candidates, truths, strict=False):
        total += 1
        try:
            rank = cand.index(truth) + 1
            rr_sum += 1.0 / rank
        except ValueError:
            continue
    return rr_sum / total if total else 0.0


def near_miss_rate(diagnostics: Sequence[dict[str, Any]]) -> float:
    """Fraction of queries marked as near misses.

    A *near miss* occurs when the top-1 retrieved item shares the query's
    ``group_id`` but the query was labelled ``should_remember=False``.
    Each diagnostics dict is expected to contain a boolean ``near_miss`` key.
    """

    if not diagnostics:
        return 0.0
    count = sum(1 for d in diagnostics if bool(d.get("near_miss")))
    return count / len(diagnostics)


def collision_rate(diagnostics: Sequence[dict[str, Any]]) -> float:
    """Fraction of queries resulting in a collision.

    A *collision* occurs when the top-1 result's ``group_id`` differs from the
    query's ``group_id`` while a matching ``should_remember=True`` item exists
    in the store. Each diagnostics dict is expected to contain a boolean
    ``collision`` key.
    """

    if not diagnostics:
        return 0.0
    count = sum(1 for d in diagnostics if bool(d.get("collision")))
    return count / len(diagnostics)


def completion_lift(baseline: Sequence[bool], hopfield: Sequence[bool]) -> float:
    """Compute lift in top-1 accuracy from Hopfield completion.

    Parameters
    ----------
    baseline:
        Sequence of top-1 correctness flags when Hopfield refinement is disabled.
    hopfield:
        Sequence of top-1 correctness flags when Hopfield refinement is enabled.

    Returns
    -------
    float
        Difference in mean top-1 correctness ``mean(hopfield) - mean(baseline)``.
        Returns ``0.0`` if sequences are empty or lengths mismatch.
    """

    if not baseline or not hopfield or len(baseline) != len(hopfield):
        return 0.0

    def _mean(xs: Sequence[bool]) -> float:
        return sum(1.0 if x else 0.0 for x in xs) / len(xs)

    return _mean(hopfield) - _mean(baseline)


def hopfield_rank_improved_rate(diagnostics: Sequence[dict[str, Any]]) -> float:
    """Fraction of queries where Hopfield improved the gold rank.

    Each diagnostics entry is expected to provide a numeric ``rank_delta`` key
    capturing the change in rank for the ground-truth ``group_id`` when
    Hopfield refinement is enabled. Positive values indicate an improvement
    (lower rank index), negative values a regression.
    """

    if not diagnostics:
        return 0.0
    total = 0
    improved = 0
    for diag in diagnostics:
        total += 1
        try:
            delta = float(diag.get("rank_delta", 0))
        except (TypeError, ValueError):
            delta = 0.0
        if delta > 0:
            improved += 1
    return improved / total if total else 0.0
