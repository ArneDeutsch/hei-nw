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
        Path("scripts/run_m2_retrieval_ci.sh"),
        Path("scripts/compare_b0_b1_m2.sh"),
    ]
    for script in scripts:
        assert script.exists(), f"{script} missing"
        assert script.stat().st_mode & 0o111, f"{script} not executable"
