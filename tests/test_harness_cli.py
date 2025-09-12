import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TINY_MODEL = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2"


def test_parse_args_default_model(tmp_path: Path) -> None:
    from hei_nw.eval.harness import DEFAULT_MODEL_ID, parse_args

    args = [
        "--mode",
        "B0",
        "--scenario",
        "A",
        "-n",
        "1",
        "--outdir",
        str(tmp_path),
    ]
    parsed = parse_args(args)
    assert parsed.model == DEFAULT_MODEL_ID


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


@pytest.mark.slow
def test_cli_rag_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "hei_nw.eval.harness",
        "--mode",
        "B0",
        "--scenario",
        "E",
        "-n",
        "4",
        "--seed",
        "0",
        "--outdir",
        str(outdir),
        "--model",
        str(TINY_MODEL),
        "--baseline",
        "rag",
    ]
    subprocess.run(cmd, check=True)  # noqa: S603
    json_files = list(outdir.glob("*_metrics.json"))
    assert json_files
    data = json.loads(json_files[0].read_text())
    assert data["compute"]["baseline"] is not None


@pytest.mark.slow
def test_cli_default_outdir(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "hei_nw.eval.harness",
        "--mode",
        "B0",
        "--scenario",
        "B",
        "-n",
        "0",
        "--seed",
        "0",
        "--model",
        str(TINY_MODEL),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)  # noqa: S603
    outdir = tmp_path / "reports" / "baseline"
    json_files = list(outdir.glob("*_metrics.json"))
    md_files = list(outdir.glob("*_report.md"))
    assert json_files and md_files


@pytest.mark.slow
def test_cli_default_outdir_b1(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "hei_nw.eval.harness",
        "--mode",
        "B1",
        "--scenario",
        "B",
        "-n",
        "0",
        "--seed",
        "0",
        "--model",
        str(TINY_MODEL),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, cwd=tmp_path, env=env, check=True)  # noqa: S603
    outdir = tmp_path / "reports" / "m1-episodic-adapter"
    json_files = list(outdir.glob("*_metrics.json"))
    md_files = list(outdir.glob("*_report.md"))
    assert json_files and md_files
