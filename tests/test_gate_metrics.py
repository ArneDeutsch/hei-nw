from __future__ import annotations

import pytest

from hei_nw.telemetry.gate import compute_gate_metrics


def test_precision_recall_computation() -> None:
    diagnostics = [
        {"score": 0.9, "should_write": True, "should_remember_label": True},
        {"score": 0.8, "should_write": True, "should_remember_label": False},
        {"score": 0.4, "should_write": False, "should_remember_label": True},
    ]
    metrics = compute_gate_metrics(diagnostics, calibration_bins=2)
    assert metrics["precision"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["recall"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["clutter_rate"] == pytest.approx(2 / 3, rel=1e-6)
    assert metrics["pr_auc"] == pytest.approx(5 / 6, rel=1e-6)
    assert metrics["writes"] == 2
    assert metrics["positives"] == 2
    assert metrics["total"] == 3
    assert len(metrics["calibration"]) == 2
    assert sum(bucket["count"] for bucket in metrics["calibration"]) == 3
