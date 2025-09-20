from __future__ import annotations

from typing import Any

import pytest

from hei_nw.eval.harness import (
    DevIsolationSettings,
    ModelGeometry,
    QAPromptSettings,
    _evaluate_mode_b1,
)


class SimpleTokenizer:
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


class HopfieldProbeService:
    def __init__(self, tokenizer: SimpleTokenizer) -> None:
        self.tokenizer = tokenizer
        self.store = self
        self.return_m = 2
        self.max_mem_tokens = 32

    def query(
        self,
        _cue: str,
        *,
        return_m: int,
        use_hopfield: bool,
        group_id: int,
        should_remember: bool,
    ) -> dict[str, Any]:  # noqa: ARG002
        truth_answer = "Alice"
        truth_entry = {
            "group_id": group_id,
            "answers": [truth_answer],
            "trace": {"group_id": group_id, "answers": [truth_answer]},
        }
        decoy_group = group_id + 100
        decoy_entry = {
            "group_id": decoy_group,
            "answers": ["decoy"],
            "trace": {"group_id": decoy_group, "answers": ["decoy"]},
        }
        baseline_candidates = [decoy_entry, truth_entry]
        baseline_diag = {
            "near_miss": False,
            "collision": False,
            "pre_top1_group": decoy_group,
            "post_top1_group": decoy_group,
            "rank_delta": 0,
        }
        if use_hopfield:
            candidates = [truth_entry, decoy_entry]
            diagnostics = {
                "near_miss": False,
                "collision": False,
                "pre_top1_group": decoy_group,
                "post_top1_group": group_id,
                "rank_delta": 1,
            }
        else:
            candidates = baseline_candidates
            diagnostics = baseline_diag
        selected = candidates[: max(1, min(return_m, len(candidates)))]
        return {
            "selected": selected,
            "candidates": candidates,
            "diagnostics": diagnostics,
            "baseline_candidates": baseline_candidates,
            "baseline_diagnostics": baseline_diag,
        }


def test_hopfield_probe_lift_non_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = SimpleTokenizer()

    def fake_generate(*_: Any, **__: Any) -> dict[str, Any]:
        return {"text": "model-output", "prompt_tokens": 2, "generated_tokens": 1}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter", lambda _model, *, scale=0.2: object()
    )

    def fake_recall_service(
        records: list[dict[str, Any]],
        tok: SimpleTokenizer,
        max_mem_tokens: int,
        **_: Any,
    ) -> HopfieldProbeService:
        assert tok is tokenizer
        assert max_mem_tokens == 128
        assert len(records) == 1
        return HopfieldProbeService(tok)

    monkeypatch.setattr("hei_nw.eval.harness.RecallService.build", fake_recall_service)

    records = [
        {
            "episode_text": "Alice met Bob at the park.",
            "cues": ["Who met Bob?"],
            "answers": ["Alice"],
            "group_id": 7,
            "should_remember": True,
            "lag": 0,
        }
    ]

    geometry = ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")
    qa_settings = QAPromptSettings(
        prompt_style="plain", max_new_tokens=4, stop=None, answer_hint=True
    )
    dev_settings = DevIsolationSettings(retrieval_only=True)

    items_no, _compute_no, _baseline_no, extra_no = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        no_hopfield=True,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=dev_settings,
    )
    assert items_no[0].prediction == "decoy (retrieval miss)"

    items_h, _compute_h, _baseline_h, extra_h = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=dev_settings,
    )
    assert items_h[0].prediction == "Alice"

    baseline_p_at_1 = float(extra_h["retrieval"]["baseline_p_at_1"])
    hopfield_p_at_1 = float(extra_h["retrieval"]["p_at_1"])
    completion_lift = float(extra_h["retrieval"]["completion_lift"])

    assert hopfield_p_at_1 >= baseline_p_at_1
    assert completion_lift > 0.0
    assert pytest.approx(baseline_p_at_1, abs=1e-6) == float(extra_no["retrieval"]["p_at_1"])
