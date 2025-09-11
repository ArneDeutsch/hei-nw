from __future__ import annotations

from hei_nw.metrics.compute import (
    ComputeRecord,
    estimate_attention_flops,
    estimate_kv_bytes,
)


def test_schema_keys_present() -> None:
    record = ComputeRecord()
    dumped = record.model_dump()
    assert "attention_flops" in dumped
    assert "kv_cache_bytes" in dumped


def test_monotonicity_tokens() -> None:
    small_flops = estimate_attention_flops(1, 1, 2, 128, 2)
    big_flops = estimate_attention_flops(2, 2, 2, 128, 2)
    assert big_flops > small_flops

    small_kv = estimate_kv_bytes(1, 128)
    big_kv = estimate_kv_bytes(2, 128)
    assert big_kv > small_kv
