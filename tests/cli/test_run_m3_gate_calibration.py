from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hei_nw.cli import run_m3_gate_calibration as m3


def test_parse_args_sets_default_sweep_when_no_threshold() -> None:
    config = m3.parse_args([])
    assert config.threshold_mode == "manual"
    assert config.threshold_values == m3.DEFAULT_THRESHOLD_SWEEP
    assert config.threshold == pytest.approx(1.5)


def test_parse_args_enables_auto_when_target() -> None:
    config = m3.parse_args(["--target", "2.0"])
    assert config.threshold_mode == "auto"
    assert not config.threshold_values


@pytest.fixture(autouse=True)
def _patch_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    def _generate(_n: int, *, seed: int) -> list[dict[str, object]]:
        return [{"gate_features": {"pin": seed % 2 == 0}}]

    monkeypatch.setattr(m3.datasets, "scenario_a", SimpleNamespace(generate=_generate))


def test_runner_auto_sweep_selects_tau(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_render_plot(_telemetry, out_path: Path, **_kwargs) -> None:
        out_path.write_text("plot", encoding="utf8")

    def fake_harness_main(args: list[str]) -> int:
        out_dir = Path(args[args.index("--outdir") + 1])
        tau = float(args[args.index("--gate.threshold") + 1])
        metric_value = 3.0 / tau
        gate = {
            "threshold": tau,
            "writes": metric_value * 10,
            "total": 10,
            "write_rate": metric_value / 10,
            "write_rate_per_1k_tokens": metric_value,
            "write_rate_per_1k_records": metric_value * 2,
            "prompt_tokens": 1000,
            "generated_tokens": 0,
            "telemetry": {
                "calibration": [
                    {
                        "lower": 0.0,
                        "upper": 1.0,
                        "count": 5,
                        "fraction_positive": 0.4,
                        "mean_score": 0.5,
                    }
                ]
            },
            "trace_samples": [],
        }
        metrics = {"dataset": {"scenario": "A"}, "gate": gate}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "A_B1_metrics.json").write_text(json.dumps(metrics), encoding="utf8")
        return 0

    monkeypatch.setattr(m3, "render_plot", fake_render_plot)
    monkeypatch.setattr(m3.harness, "main", fake_harness_main)

    config = m3.CalibrationConfig(
        scenario="A",
        n=16,
        seed=13,
        model="stub-model",
        threshold=1.5,
        threshold_provided=False,
        threshold_mode="auto",
        threshold_values=[],
        target_band=(1.0, 5.0),
        target_per="tokens",
        target_value=2.0,
        target_tolerance=0.1,
        out_dir=tmp_path,
        plot_title=None,
        pin_eval=False,
    )
    runner = m3.GateCalibrationRunner(config)
    runner.run()

    auto_meta = tmp_path / "A_auto_selected_tau.json"
    summary_json = tmp_path / "A_sweep_summary.json"
    summary_tsv = tmp_path / "A_sweep_summary.tsv"
    assert auto_meta.exists()
    assert summary_json.exists()
    assert summary_tsv.exists()

    summary = json.loads(summary_json.read_text(encoding="utf8"))
    assert summary["auto_selected_tau"] == pytest.approx(1.5)
    assert pytest.approx(summary["auto_selected_metric_value"], abs=1e-6) == 2.0
    assert any(run["threshold"] == pytest.approx(1.5) for run in summary["runs"])
