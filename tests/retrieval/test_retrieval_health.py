from __future__ import annotations

from typing import Any

import pytest

from hei_nw.eval.harness import (
    DevIsolationSettings,
    ModelGeometry,
    QAPromptSettings,
    _evaluate_mode_b1,
)


class _SimpleTokenizer:
    """Minimal tokenizer stub for retrieval tests."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, list[int]]:  # noqa: ARG002
        tokens = self.tokenize(text)
        ids = [self._vocab.setdefault(tok, len(self._vocab) + 1) for tok in tokens]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        inv_vocab = {idx: tok for tok, idx in self._vocab.items()}
        return [inv_vocab.get(tid, f"tok{tid}") for tid in token_ids]


class _RetrievalProbeService:
    def __init__(
        self,
        plan: dict[int, dict[str, Any]],
        tokenizer: _SimpleTokenizer,
        max_mem_tokens: int,
    ) -> None:
        self._plan = plan
        self.tokenizer = tokenizer
        self.store = self
        self.max_mem_tokens = max_mem_tokens
        self.return_m = 4

    def query(
        self,
        _cue: str,
        *,
        return_m: int,
        use_hopfield: bool,  # noqa: ARG002
        group_id: int,
        should_remember: bool,  # noqa: ARG002
    ) -> dict[str, Any]:
        payload = self._plan[group_id]
        truth_group = group_id
        truth_answer = payload["answer"]
        decoy_group = payload["decoy_group"]
        decoy_answer = payload["decoy_answer"]
        truth_trace = {
            "group_id": truth_group,
            "answers": [truth_answer],
            "trace": {"group_id": truth_group, "answers": [truth_answer]},
        }
        decoy_trace = {
            "group_id": decoy_group,
            "answers": [decoy_answer],
            "trace": {"group_id": decoy_group, "answers": [decoy_answer]},
        }
        if payload["top_correct"]:
            candidates = [truth_trace, decoy_trace]
        else:
            candidates = [decoy_trace, truth_trace]
        selected = candidates[: max(1, min(return_m, len(candidates)))]
        diagnostics = {"near_miss": False, "collision": False, "rank_delta": 0}
        return {
            "selected": selected,
            "candidates": candidates,
            "diagnostics": diagnostics,
            "baseline_candidates": candidates,
            "baseline_diagnostics": diagnostics,
        }


def test_p_at_1_above_0p6_on_mini(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[dict[str, Any]] = []
    plan: dict[int, dict[str, Any]] = {}
    for idx in range(10):
        records.append(
            {
                "episode_text": f"Episode {idx}",
                "cues": [f"Who answered item {idx}?"],
                "answers": [f"Answer-{idx}"],
                "group_id": idx,
                "should_remember": True,
                "lag": 0,
                "gate_features": {
                    "surprise": 1.3,
                    "novelty": 0.9,
                    "reward": False,
                    "pin": False,
                },
            }
        )
        plan[idx] = {
            "answer": f"Answer-{idx}",
            "decoy_answer": f"Alt-{idx}",
            "decoy_group": idx + 100,
            "top_correct": idx < 7,
        }

    tokenizer = _SimpleTokenizer()
    captured_kwargs: dict[str, Any] = {}

    def fake_generate(*_: Any, **__: Any) -> dict[str, Any]:
        return {"text": "Answer", "prompt_tokens": 2, "generated_tokens": 1}

    def fake_recall_service(
        recs: list[dict[str, Any]],
        tok: _SimpleTokenizer,
        max_mem_tokens: int,
        **kwargs: Any,
    ) -> _RetrievalProbeService:
        assert recs == records
        assert tok is tokenizer
        assert max_mem_tokens == 128
        captured_kwargs.update(kwargs)
        return _RetrievalProbeService(plan, tok, max_mem_tokens)

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "hei_nw.eval.harness.RecallService.build",
        fake_recall_service,
    )

    geometry = ModelGeometry(layers=4, hidden=16, heads=2, dtype="float32")
    qa_settings = QAPromptSettings(prompt_style="plain", max_new_tokens=4, stop=None, answer_hint=True)

    items, _compute, _baseline, extra = _evaluate_mode_b1(
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
        ann_m=48,
        ann_ef_construction=256,
        ann_ef_search=128,
    )

    assert items, "Expected evaluation items"
    retrieval = extra.get("retrieval", {})
    p_at_1 = float(retrieval.get("p_at_1", 0.0))
    assert p_at_1 >= 0.6
    assert captured_kwargs["ann_m"] == 48
    assert captured_kwargs["ann_ef_construction"] == 256
    assert captured_kwargs["ann_ef_search"] == 128
