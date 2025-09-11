"""Tests for long-context baseline integration in the harness."""

from pathlib import Path

from hei_nw.datasets.scenario_e import generate
from hei_nw.eval.harness import _run_baseline
from hei_nw.models.base import load_base

TINY_MODEL = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2"


def test_run_baseline_handles_scenario_e_records() -> None:
    records = generate(n=1, seed=0)
    tok, model, _ = load_base(model_id=str(TINY_MODEL), quant_4bit=False)
    compute = _run_baseline("long-context", records, model, tok)
    assert compute is not None
    assert compute["attention_flops"] is not None
    assert compute["kv_cache_bytes"] is not None
