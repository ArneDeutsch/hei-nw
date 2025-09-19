from __future__ import annotations

from hei_nw.datasets import scenario_a
from hei_nw.eval.harness import (
    DevIsolationSettings,
    ModelGeometry,
    QAPromptSettings,
    _aggregate_metrics,
    _evaluate_mode_b0,
    _evaluate_mode_b1,
    _scenario_default_qa_settings,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def __call__(self, text: str) -> dict[str, list[int]]:
        tokens = text.split()
        ids = [self._vocab.setdefault(tok, len(self._vocab) + 1) for tok in tokens]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        inverse = {idx: tok for tok, idx in self._vocab.items()}
        return [inverse.get(tid, f"tok{tid}") for tid in token_ids]


class TinyRecallService:
    def __init__(self, records: list[dict[str, object]], tokenizer: DummyTokenizer, max_mem_tokens: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_mem_tokens = max_mem_tokens
        self.return_m = 4
        self.store = self

    def query(
        self,
        _cue: str,
        *,
        return_m: int,
        use_hopfield: bool,
        group_id: int,
        should_remember: bool,
    ) -> dict[str, object]:
        answers = list(self.records[group_id]["answers"])
        candidate = {"group_id": group_id, "answers": answers}
        diagnostics = {"near_miss": False, "collision": False}
        return {"selected": [candidate], "candidates": [candidate], "diagnostics": diagnostics}


def test_m2_uplift_tiny(monkeypatch) -> None:
    records = scenario_a.generate(n=16, seed=7, hard_negative=False, confounders_ratio=0.0)
    geometry = ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")
    tokenizer = DummyTokenizer()
    qa_settings: QAPromptSettings = _scenario_default_qa_settings("A")

    def fake_generate(*_args, adapter=None, **_kwargs):
        text = "correct" if adapter is not None else "wrong"
        return {"text": text, "prompt_tokens": 4, "generated_tokens": 1}

    def fake_em(prediction: str, _truth: str) -> float:
        return 1.0 if prediction == "correct" else 0.0

    def fake_f1(prediction: str, _truth: str) -> float:
        return 1.0 if prediction == "correct" else 0.0

    def build_service(records, tokenizer, max_mem_tokens, **_kwargs):
        return TinyRecallService(records, tokenizer, max_mem_tokens)

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)
    monkeypatch.setattr("hei_nw.models.base.build_default_adapter", lambda _model, *, scale=0.2: object())
    monkeypatch.setattr("hei_nw.eval.harness.relaxed_em", fake_em)
    monkeypatch.setattr("hei_nw.eval.harness.strict_em", fake_em)
    monkeypatch.setattr("hei_nw.eval.harness.token_f1", fake_f1)
    monkeypatch.setattr("hei_nw.eval.harness.RecallService.build", build_service)

    b0_items, *_ = _evaluate_mode_b0(
        records,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        qa=qa_settings,
    )
    b1_items, *_ = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(),
        mem_max_tokens=64,
        adapter_scale=0.2,
    )

    b0_em = _aggregate_metrics(b0_items)["em"]
    b1_em = _aggregate_metrics(b1_items)["em"]
    assert b1_em >= 0.5
    assert b1_em - b0_em >= 0.30
