from __future__ import annotations

from typing import Any

import pytest

from hei_nw.datasets import scenario_a
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
        tokens = self.tokenize(text)
        ids = [self._vocab.setdefault(tok, len(self._vocab) + 1) for tok in tokens]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        inv_vocab = {idx: tok for tok, idx in self._vocab.items()}
        return [inv_vocab.get(tid, f"tok{tid}") for tid in token_ids]


class FakeRecallService:
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
        self.top_answers = {
            int(rec.get("group_id", idx)): f"{rec['answers'][0]} (retrieved)"
            for idx, rec in enumerate(records)
        }

    def query(
        self,
        cue: str,
        *,
        return_m: int,
        use_hopfield: bool,
        group_id: int,
        should_remember: bool,
    ) -> dict[str, Any]:
        answers = [self.top_answers.get(group_id, ""), "backup"]
        result = {
            "selected": [{"group_id": group_id, "answers": answers}],
            "candidates": [{"group_id": group_id, "answers": answers}],
            "diagnostics": {"near_miss": False, "collision": False},
        }
        self.calls.append({"use_hopfield": use_hopfield, "result": result})
        return result


@pytest.fixture()
def records() -> list[dict[str, Any]]:
    return scenario_a.generate(n=2, seed=0, hard_negative=False, confounders_ratio=0.0)


@pytest.fixture()
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture()
def geometry() -> ModelGeometry:
    return ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")


@pytest.fixture()
def qa_settings() -> QAPromptSettings:
    return QAPromptSettings(
        prompt_style="chat",
        max_new_tokens=16,
        stop=None,
        answer_hint=True,
        stop_mode="none",
    )


@pytest.fixture()
def stub_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate(*_: Any, **__: Any) -> dict[str, Any]:
        return {"text": "model-output", "prompt_tokens": 4, "generated_tokens": 1}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter",
        lambda _model, *, scale=0.2: object(),
    )


@pytest.fixture()
def fake_recall_service(monkeypatch: pytest.MonkeyPatch) -> list[FakeRecallService]:
    built: list[FakeRecallService] = []

    def build_service(
        records: list[dict[str, Any]],
        tokenizer: DummyTokenizer,
        max_mem_tokens: int,
        **_: Any,
    ) -> FakeRecallService:
        service = FakeRecallService(records, tokenizer, max_mem_tokens)
        built.append(service)
        return service

    monkeypatch.setattr("hei_nw.eval.harness.RecallService.build", build_service)
    return built


def test_retrieval_only_returns_top_candidate_answer(
    records: list[dict[str, Any]],
    tokenizer: DummyTokenizer,
    geometry: ModelGeometry,
    qa_settings: QAPromptSettings,
    fake_recall_service: list[FakeRecallService],
    stub_generation: None,
) -> None:
    items, compute, _baseline, extra = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(retrieval_only=True),
        mem_max_tokens=32,
    )

    assert compute.attention_flops == 0
    assert all(item.latency == 0.0 for item in items)

    service = fake_recall_service[0]
    expected = [
        service.top_answers[int(rec.get("group_id", idx))]
        for idx, rec in enumerate(records)
    ]
    predictions = [item.prediction for item in items]
    assert predictions == expected

    debug = extra.get("debug", {})
    dev_modes = debug.get("dev_modes", {})
    assert dev_modes.get("retrieval_only") is True
    assert dev_modes.get("oracle_trace") is False


def test_oracle_trace_uses_ground_truth_memory_preview(
    records: list[dict[str, Any]],
    tokenizer: DummyTokenizer,
    geometry: ModelGeometry,
    qa_settings: QAPromptSettings,
    fake_recall_service: list[FakeRecallService],
    stub_generation: None,
) -> None:
    _items, _compute, _baseline, extra = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(oracle_trace=True),
        mem_max_tokens=32,
    )

    debug = extra.get("debug", {})
    preview = debug.get("mem_preview")
    assert isinstance(preview, list) and preview

    service = fake_recall_service[0]
    expected_tokens = pack_trace(
        {
            "who": records[0]["answers"][0],
            "what": records[0]["answers"][1],
            "where": records[0]["answers"][2],
            "when": records[0]["answers"][3],
        },
        tokenizer,
        service.max_mem_tokens,
    )
    expected_preview = tokenizer.convert_ids_to_tokens(expected_tokens[:8])
    assert preview == expected_preview
    assert len(preview) <= 8

    dev_modes = debug.get("dev_modes", {})
    assert dev_modes.get("oracle_trace") is True
