from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TINY_MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "tiny-gpt2"


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
        str(TINY_MODEL),
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
