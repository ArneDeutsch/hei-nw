#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT_ROOT="${OUT:-reports/m2-k}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N="${N:-64}"
SEED="${SEED:-7}"
K_VALUES=(16 32 64 128)

if [[ -n "${K_LIST:-}" ]]; then
  # Allow callers to override the sweep via K_LIST="k1 k2 ...".
  read -r -a K_VALUES <<<"${K_LIST}"
fi

mkdir -p "$OUT_ROOT"

default_args=(
  --qa.prompt_style chat
  --qa.answer_hint
  --qa.max_new_tokens 16
  --qa.stop ''
  --qa.template_policy plain
  --hopfield.steps 2
  --hopfield.temperature 0.5
)

for k in "${K_VALUES[@]}"; do
  outdir="$OUT_ROOT/k${k}"
  mkdir -p "$outdir"
  echo "[m2-k-sweep] Running DG k=${k} (outdir: $outdir)"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" --dg.k "$k" \
    "${default_args[@]}"
done

OUT_ROOT="$OUT_ROOT" python - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
rows: list[dict[str, float]] = []
for child in sorted(out_root.iterdir()):
    if not child.is_dir() or not child.name.startswith("k"):
        continue
    metrics_path = child / "A_B1_metrics.json"
    if not metrics_path.is_file():
        continue
    data = json.loads(metrics_path.read_text(encoding="utf8"))
    aggregate = data.get("aggregate", {})
    retrieval = data.get("retrieval", {}) or {}
    records = data.get("records", [])
    non_empty = sum(1 for rec in records if str(rec.get("prediction", "")).strip())
    total = len(records)
    em = float(aggregate.get("em_relaxed", aggregate.get("em", 0.0)))
    rows.append(
        {
            "dg_k": int(child.name[1:]),
            "em_relaxed": em,
            "non_empty_rate": (non_empty / total) if total else 0.0,
            "p_at_1": float(retrieval.get("p_at_1", 0.0)),
            "mrr": float(retrieval.get("mrr", 0.0)),
        }
    )

if not rows:
    raise SystemExit("No sweep results found to summarise")
rows.sort(key=lambda row: row["dg_k"])

summary_path = out_root / "summary.csv"
with summary_path.open("w", encoding="utf8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["dg_k", "em_relaxed", "non_empty_rate", "p_at_1", "mrr"])
    writer.writeheader()
    writer.writerows(rows)

print(f"[m2-k-sweep] Wrote summary to {summary_path}")
PY
