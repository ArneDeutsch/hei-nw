from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

TINY_MODEL = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2"


def _run(tmp_path: Path, n: int) -> dict:
    outdir = tmp_path / f"out_{n}"
    cmd = [
        sys.executable,
        "-m",
        "hei_nw.eval.harness",
        "--mode",
        "B1",
        "--scenario",
        "A",
        "-n",
        str(n),
        "--seed",
        "0",
        "--outdir",
        str(outdir),
        "--model",
        str(TINY_MODEL),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603
    json_files = list(outdir.glob("*_metrics.json"))
    md_files = list(outdir.glob("*_report.md"))
    assert json_files and md_files
    data = json.loads(json_files[0].read_text())
    assert "aggregate" in data and "records" in data
    return data


def test_b1_runs_and_writes_reports(tmp_path: Path) -> None:
    data = _run(tmp_path, 2)
    retrieval = data.get("retrieval")
    assert retrieval is not None
    for v in retrieval.values():
        assert isinstance(v, int | float) and math.isfinite(float(v))
    _run(tmp_path, 0)
