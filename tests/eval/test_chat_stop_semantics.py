from typing import Any

import pytest

from hei_nw.eval.harness import (
    ModelGeometry,
    QAPromptSettings,
    _evaluate_records,
)


def _sample_records() -> list[dict[str, Any]]:
    return [
        {
            "episode_text": "Alice met Bob at the museum.",
            "cues": ["Who met Bob at the museum?"],
            "answers": ["Alice"],
            "lag": 0,
        }
    ]


def _install_generate_probe(
    monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]
) -> None:
    def fake_generate(
        prompt: Any,
        *,
        max_new_tokens: int,
        adapter: Any,
        mem_tokens: Any,
        stop: Any,
        prompt_style: str,
        stop_mode: str,
        template_policy: str,
    ) -> dict[str, Any]:
        captured.append(
            {"stop": stop, "stop_mode": stop_mode, "template_policy": template_policy}
        )
        return {"text": "Alice", "generated_tokens": 1, "prompt_tokens": 4}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)


def test_chat_stop_normalizes_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []
    _install_generate_probe(monkeypatch, captured)
    qa = QAPromptSettings(prompt_style="chat", stop="")
    geom = ModelGeometry(layers=1, hidden=1, heads=1, dtype="float32")
    items, _ = _evaluate_records(_sample_records(), geom, qa)
    assert captured == [
        {"stop": None, "stop_mode": "substring", "template_policy": "auto"}
    ]
    assert items and items[0].prediction == "Alice"


def test_chat_stop_passes_literal_newline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []
    _install_generate_probe(monkeypatch, captured)
    qa = QAPromptSettings(prompt_style="chat", stop="\n")
    geom = ModelGeometry(layers=1, hidden=1, heads=1, dtype="float32")
    _evaluate_records(_sample_records(), geom, qa)
    assert captured == [
        {"stop": "\n", "stop_mode": "substring", "template_policy": "auto"}
    ]
