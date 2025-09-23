from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, List


def _write_monotonic_stub_harness(root: Path) -> Path:
    package_root = root / "stub_pkg_monotonic"
    harness_dir = package_root / "hei_nw" / "eval"
    harness_dir.mkdir(parents=True, exist_ok=True)
    (package_root / "hei_nw" / "__init__.py").write_text("", encoding="utf8")
    (harness_dir / "__init__.py").write_text("", encoding="utf8")
    harness_code = """
import argparse
import json
from pathlib import Path


def _writes_per_1k_tokens(threshold: float) -> float:
    base = 8.0
    slope = 1.5
    value = base - slope * threshold
    if value < 0.0:
        return 0.0
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--scenario")
    parser.add_argument("-n")
    parser.add_argument("--seed")
    parser.add_argument("--model")
    parser.add_argument("--outdir")
    parser.add_argument("--gate.threshold", dest="gate_threshold")
    parser.add_argument("--eval.pins_only", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tau = float(args.gate_threshold)
    total_prompt = 500
    total_generated = 500
    total_tokens = total_prompt + total_generated
    writes_per_1k_tokens = _writes_per_1k_tokens(tau)
    writes = int(round(writes_per_1k_tokens * total_tokens / 1000.0))
    total_records = 64
    write_rate = writes / total_records if total_records else 0.0

    telemetry = {
        "writes": writes,
        "total": total_records,
        "pr_auc": 0.5,
        "writes_per_1k_tokens": writes_per_1k_tokens,
        "writes_per_1k_records": write_rate * 1000.0,
        "prompt_tokens": total_prompt,
        "generated_tokens": total_generated,
        "calibration": [
            {
                "lower": 0.0,
                "upper": 0.5,
                "count": total_records,
                "fraction_positive": write_rate,
                "mean_score": 0.25,
            }
        ],
    }
    metrics = {
        "dataset": {"scenario": args.scenario},
        "gate": {
            "threshold": tau,
            "writes": writes,
            "total": total_records,
            "write_rate": write_rate,
            "write_rate_per_1k_tokens": writes_per_1k_tokens,
            "write_rate_per_1k_records": write_rate * 1000.0,
            "prompt_tokens": total_prompt,
            "generated_tokens": total_generated,
            "telemetry": telemetry,
            "trace_samples": [],
        },
    }
    metrics_path = outdir / f"{args.scenario}_B1_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf8")


if __name__ == "__main__":
    main()
"""
    (harness_dir / "harness.py").write_text(harness_code, encoding="utf8")
    return package_root


def _assert_monotonic(values: Iterable[float]) -> None:
    iterator = iter(values)
    try:
        previous = next(iterator)
    except StopIteration:  # pragma: no cover - defensive
        return
    for current in iterator:
        assert previous >= current, f"sequence is not non-increasing: {previous} < {current}"
        previous = current


def test_monotonic_write_rate(tmp_path: Path) -> None:
    stub_root = _write_monotonic_stub_harness(tmp_path)
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    pythonpath_parts: List[str] = [str(stub_root), str(Path("src").resolve())]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    script_path = Path("scripts/run_m3_gate_calibration.sh").resolve()
    result = subprocess.run(
        [
            str(script_path),
            "--scenario",
            "A",
            "--n",
            "16",
            "--seed",
            "2",
            "--threshold-sweep",
            "0.5 1.5 2.5 3.5",
            "--out",
            str(out_dir),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    summary_path = out_dir / "A_sweep_summary.json"
    assert summary_path.exists(), "sweep summary missing"
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    runs = summary.get("runs") or []
    assert runs, "summary did not record any runs"

    thresholds: List[float] = []
    writes_per_1k_tokens: List[float] = []
    for run in runs:
        threshold = float(run["threshold"])
        thresholds.append(threshold)
        value = run.get("writes_per_1k_tokens")
        assert value is not None, f"missing writes/1k tokens for τ={threshold}"
        writes_per_1k_tokens.append(float(value))

    assert thresholds == sorted(thresholds), "thresholds were not sorted in the summary"
    _assert_monotonic(writes_per_1k_tokens)
