from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, em: float, f1: float) -> None:
    path.write_text(json.dumps({"aggregate": {"em": em, "f1": f1}}))


def test_compare_script_pass(tmp_path: Path) -> None:
    b0 = tmp_path / "A_B0_metrics.json"
    b1 = tmp_path / "A_B1_metrics.json"
    _write(b0, 0.5, 0.7)
    _write(b1, 0.55, 0.72)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/compare_b0_b1.py", str(b0), str(b1)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "| A |" in result.stdout


def test_compare_script_fail(tmp_path: Path) -> None:
    b0 = tmp_path / "A_B0_metrics.json"
    b1 = tmp_path / "A_B1_metrics.json"
    _write(b0, 0.5, 0.7)
    _write(b1, 0.2, 0.3)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/compare_b0_b1.py", str(b0), str(b1)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
