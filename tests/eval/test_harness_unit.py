import json
from pathlib import Path

import pytest

from hei_nw.eval.harness import (
    EvalItem,
    ModelGeometry,
    _aggregate_metrics,
    _hard_negative_ratio,
    _model_geometry,
    _prepare_long_context_records,
    _prepare_rag_records,
    _run_baseline,
    _save_reports,
)


class DummyCfg:
    def __init__(self) -> None:
        self.num_hidden_layers = 2
        self.hidden_size = 16
        self.num_attention_heads = 4
        self.torch_dtype = "torch.float16"


class DummyModel:
    config = DummyCfg()


def test_model_geometry_extracts_fields() -> None:
    geom = _model_geometry(DummyModel())
    assert geom == ModelGeometry(layers=2, hidden=16, heads=4, dtype="float16")


def test_aggregate_metrics_empty_and_recall() -> None:
    assert _aggregate_metrics([]) == {
        "em": 0.0,
        "f1": 0.0,
        "latency": 0.0,
        "recall_at_k": None,
    }
    items = [
        EvalItem("", "", "", 1.0, 0.5, 0.1, 0.3, 0),
        EvalItem("", "", "", 0.0, 0.5, 0.3, None, 1),
    ]
    agg = _aggregate_metrics(items)
    assert agg["em"] == pytest.approx(0.5)
    assert agg["f1"] == pytest.approx(0.5)
    assert agg["latency"] == pytest.approx(0.2)
    assert agg["recall_at_k"] == pytest.approx(0.3)


def test_prepare_long_context_records() -> None:
    gen_records = [
        {"context": "ctx", "query": "q", "expected": "a"},
        {"episode_text": "ep", "cues": ["cue"], "answers": ["ans"]},
    ]
    out = _prepare_long_context_records(gen_records)
    assert out == [
        {"context": "ctx", "query": "q", "expected": "a"},
        {"context": "ep", "query": "cue", "expected": "ans"},
    ]


def test_prepare_rag_records() -> None:
    gen_records = [
        {
            "documents": ["d1", "d2"],
            "query": "q1",
            "answers": ["a1", "a2"],
            "expected": "a1",
        },
        {"context": "ctx", "query": "q2", "expected": "e2"},
    ]
    out = _prepare_rag_records(gen_records)
    assert out[0] == {
        "documents": ["d1", "d2"],
        "query": "q1",
        "answers": ["a1", "a2"],
        "expected": "a1",
    }
    assert out[1] == {
        "documents": ["ctx"],
        "query": "q2",
        "answers": ["e2"],
        "expected": "e2",
    }


def test_run_baseline_none() -> None:
    compute, recalls = _run_baseline("none", [], object(), object())
    assert compute is None and recalls is None


def test_save_reports(tmp_path: Path) -> None:
    summary = {"foo": "bar"}
    _save_reports(tmp_path, "A", "B0", summary)
    json_files = list(tmp_path.glob("*_metrics.json"))
    md_files = list(tmp_path.glob("*_report.md"))
    assert json_files and md_files
    data = json.loads(json_files[0].read_text())
    assert data["foo"] == "bar"
    assert md_files[0].read_text().startswith("# Evaluation Report")


def test_hard_negative_ratio() -> None:
    records = [
        {"should_remember": True},
        {"should_remember": False},
        {"should_remember": False},
    ]
    assert _hard_negative_ratio("A", records) == pytest.approx(2.0)
    assert _hard_negative_ratio("B", records) is None
