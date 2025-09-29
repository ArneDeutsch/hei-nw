from __future__ import annotations

import json
from pathlib import Path

from hei_nw.cli.report_gate_write_rates import summarize_write_rates


def _write_metrics(
    path: Path,
    *,
    scenario: str,
    threshold: float,
    writes: float,
    total: float,
    write_rate: float | None = None,
    prompt_tokens: int = 1000,
    generated_tokens: int = 500,
) -> None:
    gate = {
        "threshold": threshold,
        "writes": writes,
        "total": total,
        "write_rate": write_rate,
        "write_rate_per_1k_tokens": None,
        "write_rate_per_1k_records": None,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
    }
    metrics = {"dataset": {"scenario": scenario}, "gate": gate}
    path.write_text(json.dumps(metrics), encoding="utf8")


def test_summarize_write_rates_computes_missing_fields(tmp_path: Path) -> None:
    m1 = tmp_path / "A_tau1.json"
    m2 = tmp_path / "A_tau2.json"
    _write_metrics(m1, scenario="A", threshold=1.0, writes=5, total=2000, write_rate=0.002)
    _write_metrics(m2, scenario="A", threshold=2.0, writes=3, total=2000, write_rate=0.0015)

    out_path = tmp_path / "summary.json"
    summary = summarize_write_rates(
        [m1, m2],
        out_path,
        target_band=(1.0, 5.0),
        target_per="tokens",
        auto_selected_tau=2.0,
        auto_selected_metric="writes_per_1k_tokens",
        auto_selected_metric_value=3.0,
        target_value=3.0,
    )

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf8"))
    assert data["auto_selected_tau"] == 2.0
    assert data["target_band"] == {"lower": 1.0, "upper": 5.0}
    assert len(summary["runs"]) == 2
    first = summary["runs"][0]
    assert first["scenario"] == "A"
    assert "writes_per_1k_tokens" in first
    assert first["writes_per_1k_tokens"] is not None

    second = summary["runs"][1]
    assert second["writes_per_1k_records"] is not None
