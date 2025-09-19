from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/compute_lift_ci.py").resolve()


def _write_metrics(path: Path, em_values: list[float], predictions: list[str]) -> None:
    records: list[dict[str, object]] = []
    for em, prediction in zip(em_values, predictions, strict=False):
        records.append({"em": em, "em_relaxed": em, "prediction": prediction})
    average = sum(em_values) / len(em_values)
    data = {"aggregate": {"em": average, "em_relaxed": average}, "records": records}
    path.write_text(json.dumps(data), encoding="utf8")


def test_compute_lift_ci_full(tmp_path: Path) -> None:
    b0 = tmp_path / "b0.json"
    b1 = tmp_path / "b1.json"
    _write_metrics(b0, [0.0, 0.0, 1.0], ["", "Alice", "Bob"])
    _write_metrics(b1, [1.0, 1.0, 1.0], ["Carol", "Dana", "Eve"])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(b0), str(b1), "--resamples", "100", "--seed", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Records evaluated: 3" in result.stdout
    assert "EM lift:" in result.stdout
    assert "95% bootstrap CI:" in result.stdout


def test_compute_lift_ci_subset(tmp_path: Path) -> None:
    b0 = tmp_path / "b0.json"
    b1 = tmp_path / "b1.json"
    _write_metrics(b0, [0.0, 1.0, 0.0], ["", "Alice", ""])
    _write_metrics(b1, [0.0, 1.0, 1.0], ["Bob", "Carol", "Dana"])
    subset = tmp_path / "subset.txt"
    subset.write_text("1 2\n", encoding="utf8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(b0),
            str(b1),
            "--hard-subset",
            str(subset),
            "--resamples",
            "100",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Records evaluated: 2" in result.stdout
    assert f"Hard subset: {subset}" in result.stdout
    assert "B1 non-empty rate:" in result.stdout
