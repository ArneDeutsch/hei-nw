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
    distribution = metrics["score_distribution"]
    assert isinstance(distribution, dict)
    assert distribution["p10"] <= distribution["p50"] <= distribution["p90"]
    histogram = distribution["histogram"]
    assert isinstance(histogram, list)
    assert sum(entry["count"] for entry in histogram) == 3


def test_score_distribution_fields_present() -> None:
    diagnostics = [
        {"score": 0.1, "should_write": False, "should_remember_label": False},
        {"score": 0.5, "should_write": True, "should_remember_label": True},
        {"score": 0.9, "should_write": True, "should_remember_label": True},
        {"score": 1.2, "should_write": False, "should_remember_label": False},
    ]
    metrics = compute_gate_metrics(diagnostics, calibration_bins=4)
    distribution = metrics["score_distribution"]
    assert distribution["p10"] <= distribution["p50"] <= distribution["p90"]
    histogram = distribution["histogram"]
    assert len(histogram) == 4
    assert sum(entry["count"] for entry in histogram) == len(diagnostics)


def test_pr_auc_non_trivial_when_labels_mixed() -> None:
    diagnostics = [
        {"score": 0.5, "should_write": True, "should_remember_label": True},
        {"score": 0.5, "should_write": False, "should_remember_label": False},
        {"score": 0.5, "should_write": True, "should_remember_label": False},
        {"score": 0.5, "should_write": False, "should_remember_label": True},
    ]
    metrics = compute_gate_metrics(diagnostics, calibration_bins=2)
    assert 0.0 < metrics["pr_auc"] < 1.0


def test_label_distribution_present() -> None:
    diagnostics = [
        {"score": 0.2, "should_write": False, "should_remember_label": False},
        {"score": 0.6, "should_write": True, "should_remember_label": True},
        {"score": 0.8, "should_write": True, "should_remember_label": False},
        {"score": 1.0, "should_write": False, "should_remember_label": True},
    ]
    metrics = compute_gate_metrics(diagnostics, calibration_bins=2)
    distribution = metrics.get("label_distribution")
    assert isinstance(distribution, dict)
    assert distribution["positives"] == metrics["positives"]
    assert distribution["negatives"] == metrics["total"] - metrics["positives"]
    assert 0.0 <= distribution["positive_rate"] <= 1.0
