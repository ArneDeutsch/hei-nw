#!/usr/bin/env bash
set -euo pipefail
python scripts/compare_b0_b1.py \
    reports/m2-retrieval-stack/A_B0_metrics.json \
    reports/m2-retrieval-stack/A_B1_metrics.json || true
python - <<'PY'
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _extract_em(metrics: dict[str, Any]) -> tuple[float, float]:
    aggregate = metrics.get("aggregate") or {}
    relaxed = float(aggregate.get("em_relaxed", aggregate.get("em", 0.0)))
    strict = float(aggregate.get("em_strict", aggregate.get("em", relaxed)))
    return relaxed, strict


def _extract_non_empty(metrics: dict[str, Any]) -> tuple[float, bool]:
    aggregate = metrics.get("aggregate") or {}
    rate = aggregate.get("non_empty_rate")
    if rate is not None:
        try:
            return float(rate), True
        except (TypeError, ValueError):
            pass
    records = metrics.get("records") or []
    if not records:
        return 0.0, False
    non_empty = sum(1 for record in records if str(record.get("prediction", "")).strip())
    return non_empty / len(records), False


def _extract_retrieval(metrics: dict[str, Any]) -> tuple[float | None, float | None]:
    retrieval = metrics.get("retrieval") or {}
    p_at_1 = retrieval.get("p_at_1")
    mrr = retrieval.get("mrr")
    return (
        float(p_at_1) if p_at_1 is not None else None,
        float(mrr) if mrr is not None else None,
    )


b0_path = Path("reports/m2-retrieval-stack/A_B0_metrics.json")
b1_path = Path("reports/m2-retrieval-stack/A_B1_metrics.json")
b0_metrics = _read(b0_path)
b1_metrics = _read(b1_path)
em0_relaxed, em0_strict = _extract_em(b0_metrics)
em1_relaxed, em1_strict = _extract_em(b1_metrics)
lift_relaxed = em1_relaxed - em0_relaxed
lift_strict = em1_strict - em0_strict

non_empty_rate, from_aggregate = _extract_non_empty(b1_metrics)
non_empty_source = "from aggregate" if from_aggregate else "computed from records"
print(f"B1 non-empty rate: {non_empty_rate:.3f} ({non_empty_source})")

p_at_1, mrr = _extract_retrieval(b1_metrics)
if p_at_1 is not None or mrr is not None:
    parts: list[str] = []
    if p_at_1 is not None:
        parts.append(f"P@1={p_at_1:.3f}")
    if mrr is not None:
        parts.append(f"MRR={mrr:.3f}")
    print("B1 retrieval health: " + ", ".join(parts))
else:
    print("B1 retrieval health: unavailable")

print(
    f"EM_relaxed B0={em0_relaxed:.3f} | B1={em1_relaxed:.3f}; "
    f"EM_strict B0={em0_strict:.3f} | B1={em1_strict:.3f}"
)
if lift_relaxed < 0.30:
    raise SystemExit(
        f"Relaxed EM lift {lift_relaxed:.3f} < 0.30 (strict lift {lift_strict:.3f})"
    )
print(
    f"Relaxed EM lift {lift_relaxed:.3f} >= 0.30 (strict lift {lift_strict:.3f})"
)
PY
