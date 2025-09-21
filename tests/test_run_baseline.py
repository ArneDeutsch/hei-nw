"""Tests for long-context baseline integration in the harness."""

from hei_nw.datasets.scenario_e import generate
from hei_nw.eval.harness import _run_baseline
from hei_nw.models.base import load_base
from hei_nw.testing import DUMMY_MODEL_ID


def test_run_baseline_handles_scenario_e_records() -> None:
    records = generate(n=1, seed=0)
    tok, model, _ = load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
    compute, recalls = _run_baseline("long-context", records, model, tok)
    assert compute is not None
    assert recalls is None
    assert compute["attention_flops"] is not None
    assert compute["kv_cache_bytes"] is not None


def test_run_baseline_rag_returns_recalls() -> None:
    records = generate(n=1, seed=0)
    tok, model, _ = load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
    compute, recalls = _run_baseline("rag", records, model, tok)
    assert compute is not None
    assert recalls is not None
    assert len(recalls) == len(records)
