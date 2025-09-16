from __future__ import annotations

from typing import Any

import pytest

from hei_nw.eval.harness import (
    DevIsolationSettings,
    ModelGeometry,
    QAPromptSettings,
    _evaluate_mode_b1,
)
from hei_nw.pack import pack_trace


class DummyTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def __call__(self, text: str) -> dict[str, list[int]]:
        tokens = text.split()
        ids = [self._vocab.setdefault(tok, len(self._vocab) + 1) for tok in tokens]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        inv_vocab = {idx: tok for tok, idx in self._vocab.items()}
        return [inv_vocab.get(tid, f"tok{tid}") for tid in token_ids]


@pytest.fixture()
def simple_record() -> list[dict[str, Any]]:
    return [
        {
            "episode_text": "Alice met Bob at the park.",
            "cues": ["Who met Bob?"],
            "answers": ["Alice"],
            "group_id": 1,
            "should_remember": True,
            "lag": 0,
        }
    ]


@pytest.fixture()
def geometry() -> ModelGeometry:
    return ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")


@pytest.fixture()
def qa_settings() -> QAPromptSettings:
    return QAPromptSettings(prompt_style="plain", max_new_tokens=4, stop=None, answer_hint=True)


def _patch_generation(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"count": 0}

    def fake_generate(*args: Any, **_: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"text": "model-output", "prompt_tokens": 2, "generated_tokens": 1}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)
    monkeypatch.setattr("hei_nw.models.base.build_default_adapter", lambda _model: object())
    return calls


def test_retrieval_only_uses_top_answer(
    monkeypatch: pytest.MonkeyPatch,
    simple_record: list[dict[str, Any]],
    geometry: ModelGeometry,
    qa_settings: QAPromptSettings,
) -> None:
    tokenizer = DummyTokenizer()
    call_counter = _patch_generation(monkeypatch)

    items, compute, _baseline, extra = _evaluate_mode_b1(
        simple_record,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(retrieval_only=True),
    )

    assert call_counter["count"] == len(simple_record)
    assert len(items) == len(simple_record)
    assert items[0].prediction == simple_record[0]["answers"][0]
    assert compute.attention_flops == 0
    debug = extra.get("debug", {})
    assert debug.get("dev_modes", {}).get("retrieval_only") is True
    assert debug.get("dev_modes", {}).get("oracle_trace") is False


def test_oracle_trace_injects_truthful_memory(
    monkeypatch: pytest.MonkeyPatch,
    simple_record: list[dict[str, Any]],
    geometry: ModelGeometry,
    qa_settings: QAPromptSettings,
) -> None:
    tokenizer = DummyTokenizer()
    call_counter = _patch_generation(monkeypatch)

    items, _compute, _baseline, extra = _evaluate_mode_b1(
        simple_record,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(oracle_trace=True),
    )

    assert call_counter["count"] == len(simple_record) * 2
    assert len(items) == len(simple_record)
    debug = extra.get("debug", {})
    dev_modes = debug.get("dev_modes", {})
    assert dev_modes.get("oracle_trace") is True
    mem_len = debug.get("mem_len")
    assert isinstance(mem_len, list)
    expected_tokens = pack_trace(
        {
            "who": simple_record[0]["answers"][0],
            "what": "",
            "where": "",
            "when": "",
        },
        tokenizer,
        64,
    )
    assert mem_len[0] == len(expected_tokens)
    preview = debug.get("mem_preview")
    assert preview == tokenizer.convert_ids_to_tokens(expected_tokens[:8])
