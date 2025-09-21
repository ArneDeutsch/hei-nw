from hei_nw.baselines.long_context import build_context, run_long_context
from hei_nw.models.base import load_base
from hei_nw.testing import DUMMY_MODEL_ID


def test_pack_context_not_empty() -> None:
    record = {"context": "X", "query": "Q", "expected": "X"}
    assert build_context(record)


def test_returns_compute_fields() -> None:
    tok, model, _ = load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
    records = [{"context": "A", "query": "Q", "expected": "A"}]
    out = run_long_context(model, tok, records, {"max_new_tokens": 1})
    compute = out["compute"].model_dump()
    assert compute["attention_flops"] is not None
    assert compute["kv_cache_bytes"] is not None
