#!/usr/bin/env bash
set -euo pipefail
python scripts/compare_b0_b1.py \
    reports/m2-retrieval-stack/A_B0_metrics.json \
    reports/m2-retrieval-stack/A_B1_metrics.json
python - <<'PY'
import json
with open('reports/m2-retrieval-stack/A_B0_metrics.json', 'r', encoding='utf8') as f0:
    em0 = float(json.load(f0)["aggregate"]["em"])
with open('reports/m2-retrieval-stack/A_B1_metrics.json', 'r', encoding='utf8') as f1:
    em1 = float(json.load(f1)["aggregate"]["em"])
if em1 - em0 < 0.30:
    raise SystemExit(f"EM lift {em1 - em0:.3f} < 0.30")
print(f"EM lift {em1 - em0:.3f} >= 0.30")
PY
