import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hei_nw.testing import DUMMY_MODEL_ID


@pytest.mark.slow
def test_no_hopfield_generates_ablation(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    base_cmd = [
        sys.executable,
        "-m",
        "hei_nw.eval.harness",
        "--mode",
        "B1",
        "--scenario",
        "A",
        "-n",
        "2",
        "--seed",
        "0",
        "--outdir",
        str(outdir),
        "--model",
        DUMMY_MODEL_ID,
    ]
    subprocess.run(base_cmd, check=True, env=env)  # with Hopfield
    subprocess.run(base_cmd + ["--no-hopfield"], check=True, env=env)
    png_path = outdir / "completion_ablation.png"
    assert png_path.exists()
    data = json.loads((outdir / "A_B1_no-hopfield_metrics.json").read_text())
    assert float(data["retrieval"]["completion_lift"]) >= 0
