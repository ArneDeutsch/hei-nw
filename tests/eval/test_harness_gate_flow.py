from __future__ import annotations

from typing import Any

import pytest

from hei_nw.eval.harness import (
    DevIsolationSettings,
    ModelGeometry,
    QAPromptSettings,
    _aggregate_metrics,
    _evaluate_mode_b1,
)


class _SimpleTokenizer:
    def __call__(self, text: str) -> dict[str, list[int]]:
        tokens = text.split()
        return {"input_ids": list(range(len(tokens)))}


class _StubRecallService:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._indexed = {int(rec["group_id"]): rec for rec in records}
        self.store = self
        self.return_m = 1
        self.tokenizer = _SimpleTokenizer()
        self.max_mem_tokens = 64

    def query(
        self,
        _cue: str,
        *,
        return_m: int,
        use_hopfield: bool,
        group_id: int,
        should_remember: bool,
    ) -> dict[str, Any]:  # noqa: ARG002
        record = self._indexed.get(group_id)
        diag = {"near_miss": False, "collision": False, "rank_delta": 0}
        if record is None:
            return {
                "selected": [],
                "candidates": [],
                "diagnostics": diag,
                "baseline_candidates": [],
                "baseline_diagnostics": diag,
            }
        answers = [str(ans) for ans in record.get("answers", [])]
        trace = {
            "trace_id": f"trace-{group_id}",
            "group_id": group_id,
            "answers": answers,
            "episode_text": record.get("episode_text", ""),
        }
        candidate = {"group_id": group_id, "answers": answers, "trace": trace}
        selected = [trace]
        candidates = [candidate]
        return {
            "selected": selected,
            "candidates": candidates,
            "diagnostics": diag,
            "baseline_candidates": candidates,
            "baseline_diagnostics": diag,
        }


def test_gate_metrics_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {
            "episode_text": "Alice left a notebook at the park.",
            "cues": ["Who left the notebook?"],
            "answers": ["Alice"],
            "group_id": 1,
            "should_remember": True,
            "lag": 0,
            "gate_features": {
                "surprise": 2.1,
                "novelty": 0.9,
                "reward": False,
                "pin": False,
            },
        },
        {
            "episode_text": "Ben misplaced a key at the cafe.",
            "cues": ["Who misplaced the key?"],
            "answers": ["Ben"],
            "group_id": 2,
            "should_remember": False,
            "lag": 0,
            "gate_features": {
                "surprise": 0.1,
                "novelty": 0.1,
                "reward": False,
                "pin": False,
            },
        },
    ]

    def fake_build(records: list[dict[str, Any]], *_: Any, **__: Any) -> _StubRecallService:
        return _StubRecallService(records)

    monkeypatch.setattr("hei_nw.eval.harness.RecallService.build", fake_build)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter",
        lambda _model, *, scale=0.2: object(),
    )

    geometry = ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")
    qa_settings = QAPromptSettings(
        prompt_style="plain", max_new_tokens=4, stop=None, answer_hint=True
    )

    items, _compute, _baseline, extra = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=object(),
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(retrieval_only=True),
        gate=None,
    )

    metrics = _aggregate_metrics(items)
    assert metrics["non_empty_rate"] >= 0.0

    gate_info = extra["gate"]
    assert gate_info["writes"] == 1
    assert gate_info["total"] == 2
    assert isinstance(gate_info["write_rate"], float)
    assert isinstance(gate_info["write_rate_per_1k"], float | type(None))
    telemetry = gate_info["telemetry"]
    assert telemetry["writes"] == 1
    assert telemetry["total"] == 2
    assert telemetry["calibration"]
    assert "writes_per_1k" in telemetry

    pointer_check = gate_info["pointer_check"]
    assert pointer_check["pointer_only"] is False
    assert pointer_check["missing_pointer"] >= 1
    assert "episode_text" in pointer_check["banned_keys"]

    trace_samples = gate_info.get("trace_samples")
    assert isinstance(trace_samples, list)
    assert trace_samples
    sample = trace_samples[0]
    assert sample["has_pointer"] is False
    assert "episode_text" in sample["banned_keys"]


def test_pins_only_metrics_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {
            "episode_text": "Pinned server configuration.",
            "cues": ["Which server?"],
            "answers": ["alpha"],
            "group_id": 1,
            "should_remember": True,
            "lag": 0,
            "gate_features": {
                "surprise": 2.5,
                "novelty": 0.9,
                "reward": False,
                "pin": True,
            },
        },
        {
            "episode_text": "Background status update.",
            "cues": ["What status?"],
            "answers": ["green"],
            "group_id": 2,
            "should_remember": False,
            "lag": 0,
            "gate_features": {
                "surprise": 0.1,
                "novelty": 0.1,
                "reward": False,
                "pin": False,
            },
        },
    ]

    def fake_build(records: list[dict[str, Any]], *_: Any, **__: Any) -> _StubRecallService:
        return _StubRecallService(records)

    monkeypatch.setattr("hei_nw.eval.harness.RecallService.build", fake_build)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter",
        lambda _model, *, scale=0.2: object(),
    )

    geometry = ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")
    qa_settings = QAPromptSettings(
        prompt_style="plain", max_new_tokens=4, stop=None, answer_hint=True
    )

    _, _, _, extra = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=object(),
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(retrieval_only=True),
        gate=None,
    )

    telemetry = extra["gate"]["telemetry"]
    pins_slice = telemetry.get("pins_only")
    assert pins_slice
    assert pins_slice["total"] == 1
    assert pins_slice["writes"] >= 1
    non_pins_slice = telemetry.get("non_pins")
    assert non_pins_slice
    assert non_pins_slice["total"] == 1

    _, _, _, extra_pins = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=object(),
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(retrieval_only=True),
        gate=None,
        pins_only=True,
    )

    gate_info = extra_pins["gate"]
    assert gate_info["pins_only_eval"] is True
    assert gate_info["total"] == 1
    telemetry_pins = gate_info["telemetry"]
    assert telemetry_pins["total"] == 1
    assert telemetry_pins["pins_only"]["total"] == 1
    assert telemetry_pins["non_pins"]["total"] == 1
    assert len(gate_info["decisions"]) == 1
