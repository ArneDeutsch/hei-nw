import pytest

from hei_nw.eval.report import bin_by_lag, build_markdown_report


def test_bin_by_lag_requires_two_bins() -> None:
    with pytest.raises(ValueError):
        bin_by_lag([], [0])


def test_markdown_includes_baseline_and_notes() -> None:
    summary = {
        "aggregate": {"em": 0, "f1": 0, "latency": 0},
        "lag_bins": [],
        "compute": {
            "b0": {"attention_flops": 0, "kv_cache_bytes": 0},
            "baseline": {"attention_flops": 1, "kv_cache_bytes": 2},
        },
        "dataset": {"hard_negative_ratio": 0.5},
        "adapter_latency_overhead_s": 0.012,
    }
    md = build_markdown_report(summary, scenario="A")
    assert "Baseline attention FLOPs: 1" in md
    assert "Baseline KV cache bytes: 2" in md
    assert "Hard negatives/confounders included" in md
    assert "Adapter latency overhead" in md
