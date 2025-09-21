from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hei_nw.testing import DUMMY_MODEL_ID


@pytest.mark.slow
def test_b1_summary_includes_memory_debug(tmp_path: Path) -> None:
    outdir = tmp_path / "debug"
    cmd = [
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
    subprocess.run(cmd, check=True)  # noqa: S603
    json_files = list(outdir.glob("*_metrics.json"))
    assert json_files, "expected metrics JSON file"
    data = json.loads(json_files[0].read_text())
    debug = data.get("debug")
    assert isinstance(debug, dict), "missing debug block"
    mem_len = debug.get("mem_len")
    assert isinstance(mem_len, list)
    assert all(isinstance(val, int) for val in mem_len)
    records = data.get("records")
    assert isinstance(records, list)
    assert len(mem_len) == len(records)
    mem_preview = debug.get("mem_preview")
    assert isinstance(mem_preview, list)
    assert len(mem_preview) <= 8
    assert all(isinstance(tok, str) for tok in mem_preview)
    mem_preview_str = debug.get("mem_preview_str")
    assert isinstance(mem_preview_str, str)
    first_tokens = debug.get("first_token")
    assert isinstance(first_tokens, list)
    assert len(first_tokens) == len(records)
    assert all(isinstance(tok, str) for tok in first_tokens)
