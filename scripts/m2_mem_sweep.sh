#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT_ROOT="${OUT:-reports/m2-mem-sweep}"
MODEL="${MODEL:-hei-nw/dummy-model}"
N="${N:-16}"
SEED="${SEED:-7}"
MEM_CAPS=(8 16 32 64 128)

mkdir -p "$OUT_ROOT"

run_cap() {
  local cap="$1"
  local outdir="$OUT_ROOT/mem_${cap}"
  mkdir -p "$outdir"
  echo "[m2-mem-sweep] Running mem.max_tokens=$cap (outdir: $outdir)"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" \
    --qa.prompt_style chat --qa.stop $'\n' --qa.max_new_tokens 8 --qa.answer_hint \
    --hopfield.steps 2 --hopfield.temperature 0.5 \
    --mem.max_tokens "$cap"
}

for cap in "${MEM_CAPS[@]}"; do
  run_cap "$cap"
done

python - <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
rows: list[tuple[int, float, float]] = []
for cap_dir in sorted(out_root.glob("mem_*")):
    json_path = cap_dir / "A_B1_metrics.json"
    if not json_path.exists():
        continue
    data = json.loads(json_path.read_text(encoding="utf8"))
    aggregate = data.get("aggregate", {})
    em = float(aggregate.get("em_relaxed", aggregate.get("em", 0.0)))
    latency = float(aggregate.get("latency", 0.0))
    try:
        cap_value = int(cap_dir.name.split("_")[-1])
    except ValueError:
        continue
    rows.append((cap_value, em, latency))
if not rows:
    sys.exit(0)
rows.sort()
summary_path = out_root / "summary.tsv"
with summary_path.open("w", encoding="utf8", newline="") as fh:
    writer = csv.writer(fh, delimiter="\t")
    writer.writerow(["mem_max_tokens", "em_relaxed", "latency"])
    for cap_value, em, latency in rows:
        writer.writerow([cap_value, f"{em:.6f}", f"{latency:.6f}"])
print(f"[m2-mem-sweep] Wrote summary to {summary_path}")
PY
"$OUT_ROOT"
