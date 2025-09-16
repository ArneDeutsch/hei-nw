#!/usr/bin/env bash
set -euo pipefail
python scripts/compare_b0_b1.py \
    reports/m2-retrieval-stack/A_B0_metrics.json \
    reports/m2-retrieval-stack/A_B1_metrics.json
python - <<'PY'
import json
from pathlib import Path


def _load(path: str) -> tuple[float, float]:
    data = json.loads(Path(path).read_text(encoding='utf8'))
    agg = data.get('aggregate', {})
    relaxed = float(agg.get('em_relaxed', agg.get('em', 0.0)))
    strict = float(agg.get('em_strict', agg.get('em', relaxed)))
    return relaxed, strict


em0_relaxed, em0_strict = _load('reports/m2-retrieval-stack/A_B0_metrics.json')
em1_relaxed, em1_strict = _load('reports/m2-retrieval-stack/A_B1_metrics.json')
lift_relaxed = em1_relaxed - em0_relaxed
lift_strict = em1_strict - em0_strict
if lift_relaxed < 0.30:
    raise SystemExit(
        f"Relaxed EM lift {lift_relaxed:.3f} < 0.30 (strict lift {lift_strict:.3f})"
    )
print(
    f"Relaxed EM lift {lift_relaxed:.3f} >= 0.30 (strict lift {lift_strict:.3f})"
)
print(
    f"EM_relaxed B0={em0_relaxed:.3f} | B1={em1_relaxed:.3f}; "
    f"EM_strict B0={em0_strict:.3f} | B1={em1_strict:.3f}"
)
PY
