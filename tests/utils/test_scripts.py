import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path


def test_compare_b0_b1_runs_help() -> None:
    script = Path("scripts/compare_b0_b1.py")
    subprocess.run([sys.executable, str(script), "--help"], check=True)


def test_no_stubs_regex(tmp_path: Path) -> None:
    pattern = re.compile(r"(TODO|FIXME|pass  # stub|raise NotImplementedError)")
    good = tmp_path / "good.py"
    good.write_text("print('ok')\n")
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    pass  # stub\n")
    matches = [p for p in tmp_path.rglob("*.py") if pattern.search(p.read_text())]
    assert bad in matches
    assert good not in matches


def test_m2_scripts_present_and_executable() -> None:
    scripts = [
        Path("scripts/run_m2_retrieval.sh"),
        Path("scripts/run_m2_acceptance.sh"),
        Path("scripts/run_m2_retrieval_ci.sh"),
        Path("scripts/compare_b0_b1_m2.sh"),
        Path("scripts/m2_isolation_probes.sh"),
        Path("scripts/m2_mem_sweep.sh"),
    ]
    for script in scripts:
        assert script.exists(), f"{script} missing"
        assert script.stat().st_mode & 0o111, f"{script} not executable"


def test_ci_smoke_scripts_present_and_executable() -> None:
    script = Path("scripts/ci_qa_prompting_smoke.sh")
    assert script.exists(), f"{script} missing"
    assert script.stat().st_mode & 0o111, f"{script} not executable"


def test_m3_gate_scripts_present_and_executable() -> None:
    scripts = [
        Path("scripts/run_m3_gate_calibration.sh"),
        Path("scripts/plot_gate_calibration.py"),
    ]
    for script in scripts:
        assert script.exists(), f"{script} missing"
        assert script.stat().st_mode & 0o111, f"{script} not executable"


def test_run_m2_retrieval_flags() -> None:
    script_text = Path("scripts/run_m2_retrieval.sh").read_text(encoding="utf8")
    assert "--qa.prompt_style chat" in script_text
    assert "--qa.max_new_tokens 16" in script_text
    assert "--qa.stop ''" in script_text
    assert "--hopfield.steps 2" in script_text
    assert "--hopfield.temperature 0.5" in script_text


def test_run_m2_retrieval_headroom_support() -> None:
    script_text = Path("scripts/run_m2_retrieval.sh").read_text(encoding="utf8")
    assert "Headroom Gate" in script_text
    assert "--hard-subset" in script_text


def test_m2_probe_script_includes_no_hopfield() -> None:
    script_text = Path("scripts/m2_isolation_probes.sh").read_text(encoding="utf8")
    assert script_text.count("--no-hopfield") == 1


def test_compare_b0_b1_m2_supports_bootstrap_subset() -> None:
    script_text = Path("scripts/compare_b0_b1_m2.sh").read_text(encoding="utf8")
    assert "--hard-subset" in script_text
    assert "bootstrap" in script_text


def test_run_m2_acceptance_flags() -> None:
    script_text = Path("scripts/run_m2_acceptance.sh").read_text(encoding="utf8")
    assert "--qa.max_new_tokens 16" in script_text
    assert "--qa.stop ''" in script_text
    assert "--qa.prompt_style chat" in script_text
    assert "Qwen/Qwen2.5-1.5B-Instruct" in script_text
    assert "Memory-dependent baseline" in script_text


def test_memory_dependent_baseline_helper_uses_flag() -> None:
    script_text = Path("scripts/run_m2_uplift_headroom.sh").read_text(encoding="utf8")
    assert "--qa.memory_dependent_baseline" in script_text


def test_parity_guard_uses_memory_dependent_flag() -> None:
    script_text = Path("scripts/run_parity_guard.sh").read_text(encoding="utf8")
    assert script_text.count("--qa.memory_dependent_baseline") == 2


def test_gate_non_empty_predictions_pass(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"records": [{"prediction": "Answer"}, {"prediction": "Response"}]}),
        encoding="utf8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/gate_non_empty_predictions.py")),
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_gate_non_empty_predictions_rejects_empty(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"records": [{"prediction": ""}, {"prediction": "Answer"}]}),
        encoding="utf8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/gate_non_empty_predictions.py")),
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "below threshold" in result.stderr


def test_gate_non_empty_predictions_rejects_non_alpha(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"records": [{"prediction": "123 response"}]}),
        encoding="utf8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/gate_non_empty_predictions.py")),
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "non-alphabetic first token" in result.stderr


def test_plot_gate_calibration_cli(tmp_path: Path) -> None:
    telemetry = {
        "scenario": "A",
        "threshold": 1.5,
        "calibration": [
            {"lower": 0.0, "upper": 0.5, "count": 2, "fraction_positive": 0.25, "mean_score": 0.3},
            {"lower": 0.5, "upper": 1.0, "count": 1, "fraction_positive": 0.8, "mean_score": 0.7},
        ],
    }
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry), encoding="utf8")
    out_path = tmp_path / "calibration.png"
    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/plot_gate_calibration.py")),
            str(telemetry_path),
            "--out",
            str(out_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert out_path.exists()


def test_run_m3_gate_calibration_smoke() -> None:
    script = Path("scripts/run_m3_gate_calibration.sh")
    result = subprocess.run([str(script), "--help"], capture_output=True, text=True, check=True)
    assert "Runs the B1 harness" in result.stdout


def test_run_m3_gate_calibration_has_threshold_sweep_flag() -> None:
    script_text = Path("scripts/run_m3_gate_calibration.sh").read_text(encoding="utf8")
    assert "--threshold-sweep" in script_text


def test_run_m3_gate_calibration_has_pin_eval_flag() -> None:
    script_text = Path("scripts/run_m3_gate_calibration.sh").read_text(encoding="utf8")
    assert "--pin-eval" in script_text
    assert "--eval.pins_only" in script_text


def test_threshold_sweep_creates_subdirs(tmp_path: Path) -> None:
    script = Path("scripts/run_m3_gate_calibration.sh")

    stub_root = tmp_path / "stub"
    harness_pkg = stub_root / "hei_nw" / "eval"
    harness_pkg.mkdir(parents=True, exist_ok=True)
    (stub_root / "hei_nw/__init__.py").write_text("", encoding="utf8")
    (harness_pkg / "__init__.py").write_text("", encoding="utf8")
    harness_code = textwrap.dedent(
        """
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
            parser.add_argument("--gate.threshold", dest="gate_threshold", type=float)
            parser.add_argument("--eval.pins_only", dest="eval_pins_only", action="store_true")
            args = parser.parse_args()

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            gate = {
                "threshold": args.gate_threshold,
                "writes": 2,
                "total": 10,
                "write_rate": 0.2,
                "write_rate_per_1k": 200.0,
                "pinned": 1,
                "reward_flags": 0,
                "telemetry": {
                    "pr_auc": 0.42,
                    "calibration": [
                        {
                            "lower": 0.0,
                            "upper": 0.5,
                            "count": 3,
                            "fraction_positive": 0.25,
                            "mean_score": 0.3,
                        },
                        {
                            "lower": 0.5,
                            "upper": 1.0,
                            "count": 2,
                            "fraction_positive": 0.6,
                            "mean_score": 0.7,
                        },
                    ],
                    "pins_only": {
                        "total": 1,
                        "writes": 1,
                        "precision": 1.0,
                        "recall": 1.0,
                        "pr_auc": 1.0,
                        "write_rate": 1.0,
                        "calibration": [
                            {
                                "lower": 0.5,
                                "upper": 1.0,
                                "count": 1,
                                "fraction_positive": 1.0,
                                "mean_score": 0.8,
                            }
                        ],
                    },
                    "non_pins": {
                        "total": 1,
                        "writes": 1,
                        "precision": 0.5,
                        "recall": 0.5,
                        "pr_auc": 0.5,
                        "write_rate": 0.5,
                        "calibration": [
                            {
                                "lower": 0.0,
                                "upper": 0.5,
                                "count": 1,
                                "fraction_positive": 0.25,
                                "mean_score": 0.25,
                            }
                        ],
                    },
                },
                "trace_samples": [{"index": 0, "score": 0.7}],
                "pins_only_eval": args.eval_pins_only,
            }

            gate["telemetry"]["pins_only_eval"] = args.eval_pins_only

            metrics = {
                "dataset": {"scenario": args.scenario},
                "gate": gate,
            }

            metrics_path = outdir / f"{args.scenario}_B1_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")


        if __name__ == "__main__":
            main()
        """
    )
    (harness_pkg / "harness.py").write_text(harness_code, encoding="utf8")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{stub_root}{os.pathsep}{Path('src').resolve()}"
    out_dir = tmp_path / "reports"
    env["OUT"] = str(out_dir)

    result = subprocess.run(
        [
            str(script),
            "--scenario",
            "A",
            "--n",
            "4",
            "--seed",
            "1",
            "--threshold-sweep",
            "0.9 1.1",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    for tau in ("0.9", "1.1"):
        sweep_dir = out_dir / f"tau_{tau}"
        assert sweep_dir.is_dir()

        metrics_path = sweep_dir / "A_B1_metrics.json"
        telemetry_path = sweep_dir / "A_gate_telemetry.json"
        calibration_png = sweep_dir / "A_gate_calibration.png"

        assert metrics_path.exists()
        assert telemetry_path.exists()
        assert calibration_png.exists()

        metrics_data = json.loads(metrics_path.read_text(encoding="utf8"))
        gate_data = metrics_data.get("gate", {})
        assert gate_data.get("calibration_plot")

    summary_json = out_dir / "A_sweep_summary.json"
    summary_tsv = out_dir / "A_sweep_summary.tsv"
    index_md = out_dir / "A_threshold_sweep.md"

    assert summary_json.exists()
    assert summary_tsv.exists()
    assert index_md.exists()

    summary_data = json.loads(summary_json.read_text(encoding="utf8"))
    assert len(summary_data.get("runs", [])) == 2

    tsv_lines = summary_tsv.read_text(encoding="utf8").strip().splitlines()
    assert tsv_lines[0].split("\t") == ["scenario", "tau", "write_rate", "writes_per_1k", "pr_auc"]
    assert len(tsv_lines) == 3

    index_text = index_md.read_text(encoding="utf8")
    assert "tau_0.9" in index_text
    assert "tau_1.1" in index_text


def test_pin_eval_creates_pins_outputs(tmp_path: Path) -> None:
    script = Path("scripts/run_m3_gate_calibration.sh")

    stub_root = tmp_path / "stub"
    harness_pkg = stub_root / "hei_nw" / "eval"
    datasets_pkg = stub_root / "hei_nw" / "datasets"
    harness_pkg.mkdir(parents=True, exist_ok=True)
    datasets_pkg.mkdir(parents=True, exist_ok=True)
    (stub_root / "hei_nw/__init__.py").write_text("", encoding="utf8")
    (harness_pkg / "__init__.py").write_text("", encoding="utf8")
    (datasets_pkg / "__init__.py").write_text(
        "from . import scenario_a\n__all__ = ['scenario_a']\n", encoding="utf8"
    )

    harness_code = textwrap.dedent(
        """
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
            parser.add_argument("--gate.threshold", dest="gate_threshold", type=float)
            parser.add_argument("--eval.pins_only", dest="eval_pins_only", action="store_true")
            args = parser.parse_args()

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            gate = {
                "threshold": args.gate_threshold,
                "writes": 1,
                "total": 1,
                "write_rate": 1.0,
                "write_rate_per_1k": 1000.0,
                "pinned": 1,
                "reward_flags": 0,
                "telemetry": {
                    "pr_auc": 1.0,
                    "calibration": [
                        {
                            "lower": 0.5,
                            "upper": 1.0,
                            "count": 1,
                            "fraction_positive": 1.0,
                            "mean_score": 0.8,
                        }
                    ],
                    "pins_only": {
                        "total": 1,
                        "writes": 1,
                        "precision": 1.0,
                        "recall": 1.0,
                        "pr_auc": 1.0,
                        "write_rate": 1.0,
                        "calibration": [
                            {
                                "lower": 0.5,
                                "upper": 1.0,
                                "count": 1,
                                "fraction_positive": 1.0,
                                "mean_score": 0.85,
                            }
                        ],
                    },
                    "non_pins": {
                        "total": 1,
                        "writes": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "pr_auc": 0.0,
                        "write_rate": 0.0,
                        "calibration": [
                            {
                                "lower": 0.0,
                                "upper": 0.5,
                                "count": 1,
                                "fraction_positive": 0.0,
                                "mean_score": 0.1,
                            }
                        ],
                    },
                    "pins_only_eval": args.eval_pins_only,
                },
                "pins_only_eval": args.eval_pins_only,
            }

            metrics = {
                "dataset": {"scenario": args.scenario},
                "gate": gate,
            }

            metrics_path = outdir / f"{args.scenario}_B1_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")


        if __name__ == "__main__":
            main()
        """
    )
    (harness_pkg / "harness.py").write_text(harness_code, encoding="utf8")

    dataset_code = textwrap.dedent(
        """
        from __future__ import annotations

        def generate(n: int, seed: int) -> list[dict[str, object]]:  # noqa: ARG001
            records: list[dict[str, object]] = []
            for idx in range(n):
                records.append(
                    {
                        "gate_features": {"pin": idx == 0},
                        "should_remember": idx == 0,
                    }
                )
            return records
        """
    )
    (datasets_pkg / "scenario_a.py").write_text(dataset_code, encoding="utf8")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{stub_root}{os.pathsep}{Path('src').resolve()}"
    out_dir = tmp_path / "reports"
    env["OUT"] = str(out_dir)

    result = subprocess.run(
        [
            str(script),
            "--scenario",
            "A",
            "--n",
            "2",
            "--seed",
            "1",
            "--pin-eval",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    telemetry_path = out_dir / "A_gate_telemetry_pins.json"
    calibration_path = out_dir / "A_gate_calibration_pins.png"
    assert telemetry_path.exists()
    assert calibration_path.exists()

    telemetry_data = json.loads(telemetry_path.read_text(encoding="utf8"))
    assert telemetry_data.get("pins_only_eval") is True
    assert telemetry_data.get("pins_only")
