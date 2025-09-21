import json
import re
import subprocess
import sys
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
