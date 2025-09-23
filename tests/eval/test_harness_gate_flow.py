from __future__ import annotations

from typing import Any

import pytest

from hei_nw.eval.harness import (
    DevIsolationSettings,
    ModelGeometry,
    QAPromptSettings,
    _aggregate_metrics,
    _evaluate_mode_b1,
    _summarize_gate,
)
from hei_nw.gate import NeuromodulatedGate


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
        self.ntotal = len(records)

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
    monkeypatch.setattr(
        "hei_nw.models.base.generate",
        lambda *args, **kwargs: {"text": "", "prompt_tokens": 0, "generated_tokens": 0},
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
    per_1k_tokens = gate_info["write_rate_per_1k_tokens"]
    if per_1k_tokens is not None:
        assert isinstance(per_1k_tokens, float)
    assert isinstance(gate_info["write_rate_per_1k_records"], float)
    assert gate_info["store_writes"] == 1
    assert gate_info["store_write_rate"] == pytest.approx(gate_info["write_rate"])
    assert gate_info["store_write_rate_per_1k_records"] == pytest.approx(
        gate_info["write_rate_per_1k_records"]
    )
    assert gate_info["used_for_writes"] is True
    assert gate_info["debug_keep_labels"] is False
    assert gate_info["indexed_records"] == 1
    assert gate_info["label_positive_records"] == 1
    telemetry = gate_info["telemetry"]
    assert telemetry["writes"] == 1
    assert telemetry["total"] == 2
    assert telemetry["calibration"]
    assert "writes_per_1k_tokens" in telemetry
    store_info = extra.get("store")
    assert store_info
    assert store_info["ntotal"] == 1
    assert store_info["indexed_records"] == 1

    pointer_check = gate_info["pointer_check"]
    assert pointer_check["pointer_only"] is True
    assert pointer_check["missing_pointer"] == 0
    assert pointer_check["banned_keys"] == []
    assert pointer_check["banned_key_counts"] == {}

    trace_samples = gate_info.get("trace_samples")
    assert isinstance(trace_samples, list)
    assert trace_samples
    sample = trace_samples[0]
    assert sample["has_pointer"] is True
    assert sample.get("banned_keys") == []
    pointer = sample.get("pointer")
    assert pointer
    assert pointer.get("doc")
    assert pointer.get("end") > pointer.get("start")


def test_trace_samples_are_pointer_only(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        "hei_nw.models.base.generate",
        lambda *args, **kwargs: {"text": "", "prompt_tokens": 0, "generated_tokens": 0},
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

    gate_info = extra["gate"]
    trace_samples = gate_info.get("trace_samples")
    assert isinstance(trace_samples, list)
    assert trace_samples
    for sample in trace_samples:
        assert sample["has_pointer"] is True
        assert sample.get("banned_keys") == []
        pointer = sample.get("pointer")
        assert pointer
        assert pointer.get("doc")
        assert pointer.get("end") > pointer.get("start")

    pointer_check = gate_info["pointer_check"]
    assert pointer_check["pointer_only"] is True
    assert pointer_check["missing_pointer"] == 0
    assert pointer_check["banned_key_counts"] == {}


def test_pins_only_metrics_present(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        "hei_nw.models.base.generate",
        lambda *args, **kwargs: {"text": "", "prompt_tokens": 0, "generated_tokens": 0},
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
    assert telemetry.get("pins_only_eval") is False
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
    assert telemetry_pins.get("pins_only_eval") is True
    assert telemetry_pins["total"] == 1
    assert telemetry_pins["pins_only"]["total"] == 1


def test_pin_semantics_documented(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {
            "episode_text": "Pinned episode.",
            "cues": ["What happened?"],
            "answers": ["Pinned"],
            "group_id": 7,
            "should_remember": False,
            "lag": 0,
            "gate_features": {
                "surprise": 0.0,
                "novelty": 0.0,
                "reward": False,
                "pin": True,
            },
        }
    ]

    def fake_build(records: list[dict[str, Any]], *_: Any, **__: Any) -> _StubRecallService:
        return _StubRecallService(records)

    monkeypatch.setattr("hei_nw.eval.harness.RecallService.build", fake_build)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter",
        lambda _model, *, scale=0.2: object(),
    )
    monkeypatch.setattr(
        "hei_nw.models.base.generate",
        lambda *args, **kwargs: {"text": "", "prompt_tokens": 0, "generated_tokens": 0},
    )

    geometry = ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")
    qa_settings = QAPromptSettings(
        prompt_style="plain", max_new_tokens=4, stop=None, answer_hint=True
    )

    gate = NeuromodulatedGate(threshold=3.0, pin_override=True)
    _items, _compute, _baseline, extra = _evaluate_mode_b1(
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
        gate=gate,
    )

    gate_info = extra["gate"]
    assert gate_info["pin_override"] is True
    decisions = gate_info["decisions"]
    assert decisions
    decision = decisions[0]
    assert decision["features"]["pin"] is True
    assert decision["should_write"] is True
    assert decision["indexed_for_store"] is True
    assert decision["score"] < gate.threshold
    telemetry = gate_info["telemetry"]
    assert telemetry["pins_only"]["writes"] == 1
    assert telemetry["pins_only"]["total"] == 1
    non_pin_slice = telemetry.get("non_pins")
    assert non_pin_slice is not None
    assert non_pin_slice["total"] == 0
    assert len(gate_info["decisions"]) == 1


def test_gate_controls_store_size(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {
            "episode_text": "Notable outage summary.",
            "cues": ["What happened?"],
            "answers": ["Outage"],
            "group_id": 1,
            "should_remember": True,
            "lag": 0,
            "gate_features": {
                "surprise": 2.0,
                "novelty": 0.3,
                "reward": False,
                "pin": False,
            },
        },
        {
            "episode_text": "Routine status update.",
            "cues": ["Which status?"],
            "answers": ["Green"],
            "group_id": 2,
            "should_remember": True,
            "lag": 0,
            "gate_features": {
                "surprise": 0.1,
                "novelty": 0.0,
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
    monkeypatch.setattr(
        "hei_nw.models.base.generate",
        lambda *args, **kwargs: {"text": "", "prompt_tokens": 0, "generated_tokens": 0},
    )

    geometry = ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")
    qa_settings = QAPromptSettings(
        prompt_style="plain", max_new_tokens=4, stop=None, answer_hint=True
    )

    low_threshold_gate = NeuromodulatedGate(threshold=1.0)
    _, _, _, extra_low = _evaluate_mode_b1(
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
        gate=low_threshold_gate,
        gate_use_for_writes=True,
    )

    high_threshold_gate = NeuromodulatedGate(threshold=4.0)
    _, _, _, extra_high = _evaluate_mode_b1(
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
        gate=high_threshold_gate,
        gate_use_for_writes=True,
    )

    assert extra_low["store"]["ntotal"] > extra_high["store"]["ntotal"]
    assert extra_low["gate"]["store_writes"] > extra_high["gate"]["store_writes"]

    _, _, _, extra_labels = _evaluate_mode_b1(
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
        gate=high_threshold_gate,
        gate_use_for_writes=False,
    )

    expected_labels = sum(1 for rec in records if rec.get("should_remember"))
    assert extra_labels["store"]["ntotal"] == expected_labels
    assert extra_labels["gate"]["store_writes"] == expected_labels

    _, _, _, extra_debug = _evaluate_mode_b1(
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
        gate=high_threshold_gate,
        gate_use_for_writes=True,
        gate_debug_keep_labels=True,
    )

    assert extra_debug["gate"]["store_writes"] == expected_labels
    assert extra_debug["gate"]["writes"] < expected_labels


def test_writes_per_1k_tokens_present(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {
            "episode_text": "Critical incident summary.",
            "cues": ["What incident?"],
            "answers": ["Incident"],
            "group_id": 1,
            "should_remember": True,
            "lag": 0,
            "gate_features": {
                "surprise": 2.0,
                "novelty": 0.5,
                "reward": False,
                "pin": False,
            },
        },
        {
            "episode_text": "Routine maintenance log.",
            "cues": ["What task?"],
            "answers": ["Maintenance"],
            "group_id": 2,
            "should_remember": False,
            "lag": 0,
            "gate_features": {
                "surprise": 0.1,
                "novelty": 0.0,
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

    def fake_generate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"text": "answer", "prompt_tokens": 3, "generated_tokens": 2}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)

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
        dev=DevIsolationSettings(),
        gate=NeuromodulatedGate(threshold=0.1),
    )

    gate_info = extra["gate"]
    assert gate_info["generated_tokens"] == 4
    per_1k_tokens = gate_info["write_rate_per_1k_tokens"]
    assert per_1k_tokens is not None and per_1k_tokens > 0
    telemetry = gate_info["telemetry"]
    assert telemetry["writes_per_1k_tokens"] == pytest.approx(per_1k_tokens)


def test_writes_per_1k_tokens_uses_prompt_and_generated() -> None:
    diagnostics = [
        {
            "score": 0.9,
            "should_write": True,
            "should_remember_label": True,
            "indexed_for_store": True,
            "features": {"pin": False, "reward": False},
            "prompt_tokens": 900,
            "generated_tokens": 100,
        },
        {
            "score": 0.1,
            "should_write": False,
            "should_remember_label": False,
            "indexed_for_store": False,
            "features": {"pin": False, "reward": False},
            "prompt_tokens": 100,
            "generated_tokens": 300,
        },
    ]

    gate_info = _summarize_gate(diagnostics)
    assert gate_info["prompt_tokens"] == 1000
    assert gate_info["generated_tokens"] == 400
    expected = pytest.approx(1 / ((1000 + 400) / 1000.0))
    assert gate_info["write_rate_per_1k_tokens"] == expected
    telemetry = gate_info["telemetry"]
    assert telemetry["prompt_tokens"] == 1000
    assert telemetry["generated_tokens"] == 400
    assert telemetry["writes_per_1k_tokens"] == expected
