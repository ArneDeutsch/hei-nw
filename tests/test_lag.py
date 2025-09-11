from hei_nw.eval.report import bin_by_lag


def test_bin_edges_and_counts() -> None:
    records = [
        {"lag": 0, "em": 1.0, "f1": 1.0, "recall_at_k": None},
        {"lag": 1, "em": 0.0, "f1": 0.0, "recall_at_k": None},
        {"lag": 2, "em": 1.0, "f1": 1.0, "recall_at_k": None},
    ]
    bins = [0, 1, 3]
    out = bin_by_lag(records, bins)
    assert [b["lag_bin"] for b in out] == ["0-1", "1-3"]
    assert [b["count"] for b in out] == [1, 2]
    assert out[0]["em"] == 1.0 and out[1]["em"] == 0.5
