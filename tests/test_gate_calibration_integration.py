from __future__ import annotations

from typing import Any

from hei_nw.eval.harness import _apply_gate, _summarize_gate
from hei_nw.gate import NeuromodulatedGate


def _make_record(group_id: int, surprise: float, *, include_features: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "group_id": group_id,
        "should_remember": True,
    }
    if include_features:
        record["gate_features"] = {
            "surprise": surprise,
            "novelty": 0.0,
            "reward": False,
            "pin": False,
        }
    return record


def test_tau_moves_write_rate() -> None:
    surprises = [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
    ]
    records: list[dict[str, Any]] = [
        _make_record(idx, surprise) for idx, surprise in enumerate(surprises)
    ]
    fallback_group_ids = [100, 101]
    records.extend(_make_record(group_id, 0.0, include_features=False) for group_id in fallback_group_ids)

    permissive_gate = NeuromodulatedGate(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, threshold=1.2)
    strict_gate = NeuromodulatedGate(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, threshold=2.2)

    _, permissive_diags, _ = _apply_gate(
        records,
        permissive_gate,
        use_for_writes=True,
        debug_keep_labels=False,
        allow_label_fallback=False,
    )
    permissive_summary = _summarize_gate(permissive_diags)
    permissive_rate = float(permissive_summary["write_rate_per_1k_records"])

    _, strict_diags, _ = _apply_gate(
        records,
        strict_gate,
        use_for_writes=True,
        debug_keep_labels=False,
        allow_label_fallback=False,
    )
    strict_summary = _summarize_gate(strict_diags)
    strict_rate = float(strict_summary["write_rate_per_1k_records"])

    assert permissive_rate > strict_rate
    assert strict_rate > 0.0

    _, fallback_diags, _ = _apply_gate(
        records,
        strict_gate,
        use_for_writes=True,
        debug_keep_labels=False,
        allow_label_fallback=True,
    )
    fallback_summary = _summarize_gate(fallback_diags)
    fallback_rate = float(fallback_summary["write_rate_per_1k_records"])

    fallback_written_groups = {
        diag.get("group_id")
        for diag in fallback_diags
        if diag.get("fallback_write")
    }
    assert fallback_written_groups == set(fallback_group_ids)
    assert fallback_rate > strict_rate
