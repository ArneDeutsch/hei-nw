import json
import subprocess
import sys
from pathlib import Path

import pytest

TINY_MODEL = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2"


@pytest.mark.slow
def test_cli_b0_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "hei_nw.eval.harness",
        "--mode",
        "B0",
        "--scenario",
        "A",
        "-n",
        "0",
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
