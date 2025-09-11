from hei_nw.eval.report import build_markdown_report


def test_md_contains_required_sections() -> None:
    summary = {
        "aggregate": {"em": 0.5, "f1": 0.5, "latency": 0.1},
        "lag_bins": [{"lag": 1, "count": 1, "em": 0.5, "f1": 0.5}],
        "compute": {"b0": {"attention_flops": 0, "kv_cache_bytes": 0}},
    }
    md = build_markdown_report(summary)
    assert "Aggregate Metrics" in md
    assert "Lag bins" in md
    assert "Compute" in md
