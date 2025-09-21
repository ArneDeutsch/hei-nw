import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from hei_nw.testing import DUMMY_MODEL_ID


@pytest.mark.slow
def test_b0_scenario_a_tiny(tmp_path: Path) -> None:
    modules = [
        "hei_nw",
        "hei_nw.datasets.scenario_a",
        "hei_nw.models.base",
        "hei_nw.eval.harness",
        "hei_nw.eval.report",
        "hei_nw.baselines.long_context",
    ]
    for mod in modules:
        importlib.import_module(mod)

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
        "4",
        "--seed",
        "0",
        "--outdir",
        str(outdir),
        "--model",
        DUMMY_MODEL_ID,
    ]
    subprocess.run(cmd, check=True)  # noqa: S603
    json_files = list(outdir.glob("*_metrics.json"))
    md_files = list(outdir.glob("*_report.md"))
    assert json_files and md_files
