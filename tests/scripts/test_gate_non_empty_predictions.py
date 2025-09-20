from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gate_non_empty_predictions.py"
_spec = importlib.util.spec_from_file_location("gate_non_empty_predictions", SCRIPT_PATH)
module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader is not None
_spec.loader.exec_module(module)  # type: ignore[assignment]


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ("", ""),
        ("   ", ""),
        ("Answer", "Answer"),
        (" Human: Hello", "Human:"),
        ("<nope> value", "<nope>"),
    ],
)
def test_first_token(prediction: str, expected: str) -> None:
    assert module.first_token(prediction) == expected


def test_find_invalid_first_tokens_detects_bad_prefixes() -> None:
    predictions = ["Human: hi", "Answer", " <mist> yep", "• bullet"]
    invalid = module.find_invalid_first_tokens(predictions)
    assert invalid == [
        (0, "Human:", "disallowed prefix"),
        (2, "<mist>", "disallowed prefix"),
        (3, "•", "disallowed prefix"),
    ]


def test_gate_passes_for_valid_predictions(tmp_path: Path) -> None:
    data = {"records": [{"prediction": "Answer"}, {"prediction": "Another"}]}
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(data), encoding="utf8")

    assert module.main([str(metrics_path)]) == 0


def test_gate_fails_on_invalid_first_token(tmp_path: Path) -> None:
    data = {"records": [{"prediction": "Human: hi"}, {"prediction": "Valid"}]}
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(data), encoding="utf8")

    assert module.main([str(metrics_path)]) == 1


def test_gate_fails_when_non_empty_below_threshold(tmp_path: Path) -> None:
    data = {"records": [{"prediction": ""}, {"prediction": "Valid"}]}
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(data), encoding="utf8")

    assert module.main([str(metrics_path), "--threshold", "0.9"]) == 1
