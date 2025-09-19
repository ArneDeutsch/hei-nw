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
