from __future__ import annotations

from typing import Any

import pytest

from hei_nw.eval.harness import (
    DevIsolationSettings,
    ModelGeometry,
    QAPromptSettings,
    _evaluate_mode_b1,
)


class DummyTokenizer:
    """Minimal tokenizer stub mirroring the behaviour used in harness tests."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def __call__(self, text: str) -> dict[str, list[int]]:
        tokens = self.tokenize(text)
        ids = [self._vocab.setdefault(tok, len(self._vocab) + 1) for tok in tokens]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        inv_vocab = {idx: tok for tok, idx in self._vocab.items()}
        return [inv_vocab.get(tid, f"tok{tid}") for tid in token_ids]


class RecordingRecallService:
    """Recall service stub capturing query invocations."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: DummyTokenizer,
        max_mem_tokens: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_mem_tokens = max_mem_tokens
        self.return_m = 4
        self.store = self
        self.calls: list[dict[str, Any]] = []

    def query(
        self,
        cue: str,
        *,
        return_m: int,
        use_hopfield: bool,
        group_id: int,
        should_remember: bool,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "cue": cue,
                "return_m": return_m,
                "use_hopfield": use_hopfield,
                "group_id": group_id,
                "should_remember": should_remember,
            }
        )
        return {
            "selected": [],
            "candidates": [{"group_id": group_id}],
            "diagnostics": {},
            "baseline_candidates": [],
            "baseline_diagnostics": {},
        }


@pytest.fixture()
def oracle_records() -> list[dict[str, Any]]:
    return [
        {
            "episode_text": "Alice met Bob at the park.",
            "cues": ["Who met Bob?"],
            "answers": ["Alice", ""],
            "group_id": 0,
            "should_remember": True,
            "lag": 0,
        },
        {
            "episode_text": "Carol delivered the parcel on Monday.",
            "cues": ["Who delivered the parcel?"],
            "answers": ["Carol", ""],
            "group_id": 1,
            "should_remember": True,
            "lag": 0,
        },
        {
            "episode_text": "Eli spotted the comet in June.",
            "cues": ["Who spotted the comet?"],
            "answers": ["Eli", ""],
            "group_id": 2,
            "should_remember": True,
            "lag": 0,
        },
    ]


@pytest.fixture()
def geometry() -> ModelGeometry:
    return ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")


@pytest.fixture()
def qa_settings() -> QAPromptSettings:
    return QAPromptSettings(
        prompt_style="chat",
        max_new_tokens=8,
        stop=None,
        answer_hint=True,
        stop_mode="none",
    )


@pytest.fixture()
def fake_recall_service(
    monkeypatch: pytest.MonkeyPatch, oracle_records: list[dict[str, Any]]
) -> list[RecordingRecallService]:
    built: list[RecordingRecallService] = []

    def build_service(
        records: list[dict[str, Any]],
        tokenizer: DummyTokenizer,
        max_mem_tokens: int,
        **_: Any,
    ) -> RecordingRecallService:
        service = RecordingRecallService(records, tokenizer, max_mem_tokens)
        built.append(service)
        return service

    monkeypatch.setattr("hei_nw.eval.harness.RecallService.build", build_service)
    return built


def test_oracle_probe_has_high_em(
    monkeypatch: pytest.MonkeyPatch,
    oracle_records: list[dict[str, Any]],
    geometry: ModelGeometry,
    qa_settings: QAPromptSettings,
    fake_recall_service: list[RecordingRecallService],
) -> None:
    answers = [rec["answers"][0] for rec in oracle_records]
    mem_call_index = {"value": 0}
    call_log: list[dict[str, Any]] = []

    def fake_generate(*_: Any, **kwargs: Any) -> dict[str, Any]:
        call_log.append({"mem_tokens": kwargs.get("mem_tokens"), "memory_prompt": kwargs.get("memory_prompt")})
        mem_tokens = kwargs.get("mem_tokens") or []
        if mem_tokens:
            memory_prompt = kwargs.get("memory_prompt")
            assert isinstance(memory_prompt, str) and memory_prompt
            idx = mem_call_index["value"]
            mem_call_index["value"] = idx + 1
            text = answers[idx]
        else:
            text = ""
        return {"text": text, "prompt_tokens": 4, "generated_tokens": 1}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter",
        lambda _model, *, scale=0.2: object(),
    )

    tokenizer = DummyTokenizer()
    items, compute, _baseline, extra = _evaluate_mode_b1(
        oracle_records,
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

    mem_calls = [entry for entry in call_log if entry["mem_tokens"]]
    assert len(mem_calls) == len(oracle_records)
    assert all(entry["memory_prompt"] for entry in mem_calls)

    assert items, "expected B1 evaluation items"
    em = sum(item.em_relaxed for item in items) / len(items)
    assert em >= 0.8
    assert all(item.prediction == item.truth for item in items)

    debug = extra.get("debug", {})
    mem_lengths = debug.get("mem_len")
    assert isinstance(mem_lengths, list)
    assert all(length > 0 for length in mem_lengths)


def test_selected_contains_answers(
    monkeypatch: pytest.MonkeyPatch,
    oracle_records: list[dict[str, Any]],
    geometry: ModelGeometry,
    qa_settings: QAPromptSettings,
    fake_recall_service: list[RecordingRecallService],
) -> None:
    def fake_generate(*_: Any, **__: Any) -> dict[str, Any]:
        return {"text": "", "prompt_tokens": 4, "generated_tokens": 1}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter",
        lambda _model, *, scale=0.2: object(),
    )

    tokenizer = DummyTokenizer()
    _items, _compute, _baseline, extra = _evaluate_mode_b1(
        oracle_records,
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

    debug = extra.get("debug", {})
    preview_str = debug.get("mem_preview_str", "")
    assert oracle_records[0]["answers"][0] in preview_str
    dev_modes = debug.get("dev_modes", {})
    assert dev_modes.get("oracle_trace") is True
