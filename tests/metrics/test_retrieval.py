import pytest

from hei_nw.metrics.retrieval import (
    collision_rate,
    completion_lift,
    hopfield_rank_improved_rate,
    mrr,
    near_miss_rate,
    precision_at_k,
)


def test_p_at_k() -> None:
    candidates = [[1, 2, 3], [4, 5, 6]]
    truths = [1, 6]
    assert precision_at_k(candidates, truths, k=2) == 0.5


def test_mrr() -> None:
    candidates = [[2, 1], [3, 4, 5]]
    truths = [1, 5]
    expected = 0.5 + 1 / 3
    assert mrr(candidates, truths) == expected / 2


def test_near_miss_and_collision() -> None:
    diagnostics = [
        {"near_miss": True, "collision": False},
        {"near_miss": False, "collision": True},
        {"near_miss": False, "collision": False},
    ]
    assert near_miss_rate(diagnostics) == 1 / 3
    assert collision_rate(diagnostics) == 1 / 3


def test_completion_lift() -> None:
    baseline = [True, False, False]
    hopfield = [True, True, False]
    assert completion_lift(baseline, hopfield) == (2 / 3) - (1 / 3)


def test_hopfield_rank_improved_rate() -> None:
    diagnostics = [
        {"rank_delta": 2},  # Improved from rank 3 to 1
        {"rank_delta": 0},  # Stayed at the same rank
        {"rank_delta": -1},  # Hopfield made the rank worse
    ]
    assert hopfield_rank_improved_rate(diagnostics) == pytest.approx(1 / 3)
