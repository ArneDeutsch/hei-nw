from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_predictions(path: Path, predictions: list[str]) -> None:
    data = {"records": [{"prediction": value} for value in predictions]}
    path.write_text(json.dumps(data))


def test_gate_passes_when_rate_high(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    _write_predictions(metrics, ["answer"] * 9 + [""])
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/gate_non_empty_predictions.py",
            str(metrics),
            "--threshold",
            "0.9",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Non-empty rate:" in result.stdout


def test_gate_fails_when_rate_low(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    _write_predictions(metrics, ["answer", "", "  ", "\t", "bar"])
    result = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/gate_non_empty_predictions.py", str(metrics)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "below threshold" in result.stderr
