from __future__ import annotations

from hei_nw.metrics.text import exact_match, recall_at_k, token_f1


def test_em_and_f1_basic() -> None:
    assert exact_match("hello", "hello") == 1.0
    assert exact_match("hello", "world") == 0.0
    assert token_f1("a b", "a c") == 0.5


def test_recall_at_k_shape() -> None:
    candidates = ["b", "a", "c"]
    truths = ["b", "d"]
    assert recall_at_k(candidates, truths, k=1) == 0.5
    assert recall_at_k(candidates, truths, k=2) == 0.5
    assert recall_at_k(candidates, truths, k=10) == 0.5
