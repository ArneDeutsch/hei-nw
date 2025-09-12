from hei_nw.metrics.text import recall_at_k, token_f1


def test_token_f1_edge_cases() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("a", "") == 0.0
    assert token_f1("", "a") == 0.0
    assert token_f1("a b", "b a") == token_f1("b a", "a b")


def test_recall_at_k_edges() -> None:
    assert recall_at_k([], ["a"], 0) == 0.0
    assert recall_at_k(["a"], [], 1) == 0.0
    assert recall_at_k(["a", "b"], ["c"], -1) == 0.0
