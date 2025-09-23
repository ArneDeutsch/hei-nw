from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _write_stub_harness(root: Path) -> Path:
    package_root = root / "stub_pkg"
    harness_dir = package_root / "hei_nw" / "eval"
    harness_dir.mkdir(parents=True, exist_ok=True)
    (package_root / "hei_nw" / "__init__.py").write_text("", encoding="utf8")
    (harness_dir / "__init__.py").write_text("", encoding="utf8")
    harness_code = """
import argparse
import json
from pathlib import Path


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
    total_prompt = 2000
    total_generated = 2000
    total_tokens = total_prompt + total_generated
    writes_per_1k = 15.0 / (tau + 0.5)
    writes = max(1, int(round(writes_per_1k * total_tokens / 1000.0)))
    total_records = 32
    write_rate = writes / total_records
    telemetry = {
        "writes": writes,
        "total": total_records,
        "pr_auc": 0.5,
        "writes_per_1k_tokens": writes_per_1k,
        "writes_per_1k_records": write_rate * 1000.0,
        "calibration": [
            {
                "lower": 0.0,
                "upper": 0.5,
                "count": 5,
                "fraction_positive": 0.2,
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
            "write_rate_per_1k_tokens": writes_per_1k,
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


def test_auto_sweep_brackets_target(tmp_path: Path) -> None:
    stub_root = _write_stub_harness(tmp_path)
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    pythonpath_parts = [str(stub_root), str(Path("src").resolve())]
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
            "1",
            "--threshold-sweep",
            "auto",
            "--target-band",
            "1,5",
            "--target-per",
            "tokens",
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
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    assert summary.get("target_band") == {"lower": 1.0, "upper": 5.0}
    assert summary.get("target_per") == "tokens"
    first_tau = summary.get("first_tau_within_target_band")
    assert first_tau is not None
    auto_selected_tau = summary.get("auto_selected_tau")
    assert auto_selected_tau is not None
    assert auto_selected_tau == first_tau

    runs = summary.get("runs") or []
    assert len(runs) >= 2
    matching = [run for run in runs if run.get("threshold") == first_tau]
    assert matching, f"no run matched τ={first_tau!r}"
    run = matching[0]
    writes_per_1k = float(run["writes_per_1k_tokens"])
    assert 1.0 <= writes_per_1k <= 5.0
