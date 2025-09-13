from pathlib import Path


def test_run_m2_defaults_to_qwen() -> None:
    text = Path("scripts/run_m2_retrieval.sh").read_text()
    assert 'MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"' in text


def test_ci_script_pins_tiny() -> None:
    text = Path("scripts/run_m2_retrieval_ci.sh").read_text()
    assert 'MODEL="tests/models/tiny-gpt2"' in text
