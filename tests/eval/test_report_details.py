from pathlib import Path

import pytest

from hei_nw.eval.report import (
    bin_by_lag,
    build_markdown_report,
    save_completion_ablation_plot,
)


def test_bin_by_lag_requires_two_bins() -> None:
    with pytest.raises(ValueError):
        bin_by_lag([], [0])


def test_markdown_includes_baseline_and_notes() -> None:
    summary = {
        "aggregate": {
            "em": 0,
            "em_relaxed": 0,
            "em_strict": 0,
            "f1": 0,
            "latency": 0,
        },
        "lag_bins": [],
        "compute": {
            "b0": {"attention_flops": 0, "kv_cache_bytes": 0},
            "baseline": {"attention_flops": 1, "kv_cache_bytes": 2},
        },
        "dataset": {"hard_negative_ratio": 0.5},
        "adapter_latency_overhead_s": 0.012,
        "retrieval": {
            "p_at_1": 0.5,
            "mrr": 0.5,
            "near_miss_rate": 0.0,
            "collision_rate": 0.0,
            "completion_lift": 0.1,
        },
    }
    md = build_markdown_report(summary, scenario="A")
    assert "Baseline attention FLOPs: 1" in md
    assert "Baseline KV cache bytes: 2" in md
    assert "Hard negatives/confounders included" in md
    assert "Adapter latency overhead" in md
    assert "P@1" in md
    assert "Completion lift" in md
    debug_section = md.split("## Debug", maxsplit=1)[1]
    assert "- None" in debug_section.split("## Dataset", maxsplit=1)[0]


def test_ablation_plot_written(tmp_path: Path) -> None:
    with_hp = {"retrieval": {"completion_lift": 0.2}}
    without_hp = {"retrieval": {"completion_lift": 0.0}}
    path = save_completion_ablation_plot(tmp_path, with_hp, without_hp)
    assert path.exists()
