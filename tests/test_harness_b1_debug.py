from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hei_nw.testing import DUMMY_MODEL_ID


@pytest.mark.slow
def test_mem_len_respects_cli_cap(tmp_path: Path) -> None:
    outdir = tmp_path / "mem_cap"
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
        "--mem.max_tokens",
        "32",
    ]
    subprocess.run(cmd, check=True)  # noqa: S603
    json_files = list(outdir.glob("*_metrics.json"))
    assert json_files, "expected metrics JSON file"
    data = json.loads(json_files[0].read_text())
    debug = data.get("debug")
    assert isinstance(debug, dict), "missing debug block"
    mem_len = debug.get("mem_len")
    assert isinstance(mem_len, list) and mem_len, "mem_len should be populated"
    assert all(isinstance(val, int) and val <= 32 for val in mem_len)
    mem_preview_str = debug.get("mem_preview_str")
    assert isinstance(mem_preview_str, str)
    first_tokens = debug.get("first_token")
    assert isinstance(first_tokens, list)
