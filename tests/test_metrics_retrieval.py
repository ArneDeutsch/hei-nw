from __future__ import annotations

import math

import pytest

from hei_nw.metrics.retrieval import (
    collision_rate,
    completion_lift,
    mrr,
    near_miss_rate,
    precision_at_k,
)


def test_precision_at_k_monotonic_and_bounded() -> None:
    candidates = [
        [3, 1, 2],
        [2, 3, 1],
    ]
    truths = [1, 3]
    p_at_1 = precision_at_k(candidates, truths, k=1)
    p_at_2 = precision_at_k(candidates, truths, k=2)
    p_at_3 = precision_at_k(candidates, truths, k=3)
    assert 0.0 <= p_at_1 <= 1.0
    assert p_at_1 <= p_at_2 <= p_at_3 <= 1.0


def test_mrr_single_hit_matches_reciprocal_rank() -> None:
    candidates = [[42, 7, 99]]
    truths = [7]
    assert math.isclose(mrr(candidates, truths), 1.0 / 2.0)


def test_near_miss_and_collision_aggregate_booleans() -> None:
    diagnostics = [
        {"near_miss": True, "collision": False},
        {"near_miss": False, "collision": True},
        {"near_miss": False, "collision": False},
    ]
    assert near_miss_rate(diagnostics) == pytest.approx(1 / 3)
    assert collision_rate(diagnostics) == pytest.approx(1 / 3)


def test_completion_lift_behaviour() -> None:
    assert completion_lift([], []) == 0.0
    assert completion_lift([True], []) == 0.0
    assert completion_lift([True, False], [True]) == 0.0
    baseline = [True, False, False]
    hopfield = [True, True, False]
    expected = (2 / 3) - (1 / 3)
    assert completion_lift(baseline, hopfield) == pytest.approx(expected)
