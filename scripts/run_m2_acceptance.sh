#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT="${OUT:-reports/m2-acceptance}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N="${N:-24}"
SEED="${SEED:-7}"
CI_SAMPLES="${CI_SAMPLES:-1000}"
HARD_SUBSET=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_m2_acceptance.sh [--hard-subset PATH]

Runs the end-to-end M2 acceptance workflow:
  • Parity guard (B1 empty ≈ B0).
  • Isolation probes (E0–E3).
  • Headroom gate on the standard Scenario A config.
  • Memory-dependent uplift run with bootstrap CI.
  • Summary report at reports/m2-acceptance/summary.md.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hard-subset)
      HARD_SUBSET="${2:-}"
      shift 2
      ;;
    --hard-subset=*)
      HARD_SUBSET="${1#*=}"
      shift 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT"

log_step() {
  echo "[m2-acceptance] $1"
}

log_step "Running parity guard"
MODEL="$MODEL" N="$N" SEED="$SEED" bash scripts/run_parity_guard.sh

log_step "Running isolation probes"
PROBES_OUT="$OUT/probes"
mkdir -p "$PROBES_OUT"
MODEL="$MODEL" OUT="$PROBES_OUT" N="$N" SEED="$SEED" \
  scripts/m2_isolation_probes.sh --probe ALL | tee "$OUT/probes_summary.txt"

log_step "Evaluating headroom on Scenario A"
RETR_OUT="$OUT/retrieval"
mkdir -p "$RETR_OUT"

default_args=(
  --qa.prompt_style chat
  --qa.answer_hint
  --qa.max_new_tokens 16
  --qa.stop ''
  --hopfield.steps 2
  --hopfield.temperature 0.5
  --gate.threshold "${GATE_THRESHOLD:-1.5}"
)

gate_flag="--no-gate.use_for_writes"
if [[ "${USE_GATE_WRITES:-0}" -ne 0 ]]; then
  gate_flag="--gate.use_for_writes"
fi

run_harness() {
  local mode="$1"
  shift
  python -m hei_nw.eval.harness --mode "$mode" --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$RETR_OUT" \
    "$gate_flag" "${default_args[@]}" "$@"
}

b1_template=(--qa.template_policy plain)

run_harness B0
run_harness B1 "${b1_template[@]}"
python scripts/gate_non_empty_predictions.py "$RETR_OUT/A_B1_metrics.json"
run_harness B1 --no-hopfield "${b1_template[@]}"

headroom_info=$(python - <<PY
from __future__ import annotations
import json
from pathlib import Path

metrics = json.loads(Path("$RETR_OUT/A_B0_metrics.json").read_text(encoding="utf8"))
agg = metrics.get("aggregate", {})
em = float(agg.get("em_relaxed", agg.get("em", 0.0)))
threshold = 0.7
status = "PASS" if em < threshold else "BLOCKED"
print(f"{status}|{em:.3f}|{threshold:.3f}")
PY
)
IFS='|' read -r HEADROOM_STATUS HEADROOM_EM HEADROOM_THRESHOLD <<<"$headroom_info"
log_step "Headroom Gate: $HEADROOM_STATUS (EM_B0=${HEADROOM_EM}, threshold=${HEADROOM_THRESHOLD})"

MEM_OUT="$OUT/memory_dependent"
SUMMARY_PATH="$OUT/summary.md"

if [[ "$HEADROOM_STATUS" == "BLOCKED" ]]; then
  log_step "Running memory-dependent baseline"
  mkdir -p "$MEM_OUT"
  MODEL="$MODEL" OUT="$MEM_OUT" N="$N" SEED="$SEED" scripts/run_m2_uplift_headroom.sh
  python scripts/gate_non_empty_predictions.py "$MEM_OUT/A_B1_metrics.json"
  OUT="$OUT" SUMMARY_PATH="$SUMMARY_PATH" HEADROOM_THRESHOLD="$HEADROOM_THRESHOLD" python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT"])
retr_dir = out_dir / "retrieval"
mem_dir = out_dir / "memory_dependent"
summary_path = Path(os.environ.get("SUMMARY_PATH", str(out_dir / "summary.md")))
summary_path.parent.mkdir(parents=True, exist_ok=True)

def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf8"))

def em_and_rate(metrics: dict[str, object]) -> tuple[float, float]:
    records = metrics.get("records", [])  # type: ignore[assignment]
    non_empty = sum(1 for rec in records if str(rec.get("prediction", "")).strip())  # type: ignore[arg-type]
    total = len(records)  # type: ignore[arg-type]
    aggregate = metrics.get("aggregate", {})  # type: ignore[assignment]
    em = float(aggregate.get("em_relaxed", aggregate.get("em", 0.0)))
    rate = non_empty / total if total else 0.0
    return em, rate

b0 = load_json(retr_dir / "A_B0_metrics.json")
b1 = load_json(retr_dir / "A_B1_metrics.json")
b1_no = load_json(retr_dir / "A_B1_no-hopfield_metrics.json")
mem_b0 = load_json(mem_dir / "A_B0_metrics.json")
mem_b1 = load_json(mem_dir / "A_B1_metrics.json")

em_b0, rate_b0 = em_and_rate(b0)
em_b1, rate_b1 = em_and_rate(b1)
em_b1_no, _ = em_and_rate(b1_no)
mem_em_b0, mem_rate_b0 = em_and_rate(mem_b0)
mem_em_b1, mem_rate_b1 = em_and_rate(mem_b1)
hopfield_delta = em_b1 - em_b1_no

with summary_path.open("w", encoding="utf8") as fh:
    fh.write("# M2 Acceptance Summary\n\n")
    fh.write("**Headroom Gate:** BLOCKED (EM_B0 = {:.3f} ≥ {}).\\n\n".format(
        em_b0, os.environ["HEADROOM_THRESHOLD"]
    ))
    fh.write("**Retrieval (chat prompt)**\\n")
    fh.write(f"- EM_B0: {em_b0:.3f}\n")
    fh.write(f"- Non-empty rate (B0): {rate_b0:.3f}\n")
    fh.write(f"- EM_B1: {em_b1:.3f}\n")
    fh.write(f"- EM_B1 (no Hopfield): {em_b1_no:.3f}\n")
    fh.write(f"- Hopfield lift: {hopfield_delta:+.3f}\n")
    fh.write(f"- Non-empty rate (B1): {rate_b1:.3f}\n")
    fh.write("\n**Memory-dependent baseline**\\n")
    fh.write(f"- EM_B0 (no episode prompt): {mem_em_b0:.3f}\n")
    fh.write(f"- Non-empty rate (B0): {mem_rate_b0:.3f}\n")
    fh.write(f"- EM_B1 (with memory tokens): {mem_em_b1:.3f}\n")
    fh.write(f"- Non-empty rate (B1): {mem_rate_b1:.3f}\n")
    fh.write("- Uplift: skipped (headroom blocked)\n")
    fh.write("\nSee probes summary at reports/m2-acceptance/probes_summary.txt.\n")
PY
  exit 0
fi

log_step "Running memory-dependent baseline"
mkdir -p "$MEM_OUT"
MODEL="$MODEL" OUT="$MEM_OUT" N="$N" SEED="$SEED" scripts/run_m2_uplift_headroom.sh
python scripts/gate_non_empty_predictions.py "$MEM_OUT/A_B1_metrics.json"

log_step "Computing uplift and bootstrap CI"
compare_args=()
if [[ -n "$HARD_SUBSET" ]]; then
  compare_args+=("--hard-subset" "$HARD_SUBSET")
fi
CI_SAMPLES="$CI_SAMPLES" OUTDIR="$MEM_OUT" scripts/compare_b0_b1_m2.sh "${compare_args[@]}" | tee "$OUT/uplift_compare.txt"

OUT="$OUT" SUMMARY_PATH="$SUMMARY_PATH" HEADROOM_THRESHOLD="$HEADROOM_THRESHOLD" CI_SAMPLES="$CI_SAMPLES" python - <<'PY'
from __future__ import annotations

import json
import math
import random
import os
from pathlib import Path

out_dir = Path(os.environ["OUT"])
retr_dir = out_dir / "retrieval"
mem_dir = out_dir / "memory_dependent"
summary_path = Path(os.environ.get("SUMMARY_PATH", str(out_dir / "summary.md")))
summary_path.parent.mkdir(parents=True, exist_ok=True)

rng = random.Random(0)

def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf8"))

def em_and_rate(metrics: dict[str, object]) -> tuple[float, float]:
    records = metrics.get("records", [])  # type: ignore[assignment]
    non_empty = sum(1 for rec in records if str(rec.get("prediction", "")).strip())  # type: ignore[arg-type]
    total = len(records)  # type: ignore[arg-type]
    aggregate = metrics.get("aggregate", {})  # type: ignore[assignment]
    em = float(aggregate.get("em_relaxed", aggregate.get("em", 0.0)))
    rate = non_empty / total if total else 0.0
    return em, rate

def em_records(metrics: dict[str, object]) -> list[float]:
    records = metrics.get("records", [])  # type: ignore[assignment]
    return [
        float(rec.get("em_relaxed", rec.get("em", 0.0)))  # type: ignore[arg-type]
        for rec in records
    ]

b0 = load_json(retr_dir / "A_B0_metrics.json")
b1 = load_json(retr_dir / "A_B1_metrics.json")
b1_no = load_json(retr_dir / "A_B1_no-hopfield_metrics.json")
mem_b0 = load_json(mem_dir / "A_B0_metrics.json")
mem_b1 = load_json(mem_dir / "A_B1_metrics.json")

em_b0, rate_b0 = em_and_rate(b0)
em_b1, rate_b1 = em_and_rate(b1)
em_b1_no, _ = em_and_rate(b1_no)
hopfield_delta = em_b1 - em_b1_no

retrieval = b1.get("retrieval", {})  # type: ignore[assignment]
p_at_1 = retrieval.get("p_at_1")
mrr = retrieval.get("mrr")
near_miss = retrieval.get("near_miss_rate")
collision = retrieval.get("collision_rate")
completion_lift = retrieval.get("completion_lift")

mem_em_b0, mem_rate_b0 = em_and_rate(mem_b0)
mem_em_b1, mem_rate_b1 = em_and_rate(mem_b1)

records_b0 = em_records(mem_b0)
records_b1 = em_records(mem_b1)
if len(records_b0) != len(records_b1):
    raise SystemExit("Memory-dependent B0/B1 record count mismatch")

lift = sum(records_b1) / len(records_b1) - sum(records_b0) / len(records_b0)

ci_samples = int(os.environ["CI_SAMPLES"])
boots: list[float] = []
indices = list(range(len(records_b0)))
for _ in range(ci_samples):
    sample = [rng.choice(indices) for _ in indices]
    b0_avg = sum(records_b0[i] for i in sample) / len(sample)
    b1_avg = sum(records_b1[i] for i in sample) / len(sample)
    boots.append(b1_avg - b0_avg)
boots.sort()
lo_idx = max(0, math.floor(0.025 * (ci_samples - 1)))
hi_idx = min(len(boots) - 1, math.ceil(0.975 * (ci_samples - 1)))
ci_low = boots[lo_idx]
ci_high = boots[hi_idx]

with summary_path.open("w", encoding="utf8") as fh:
    fh.write("# M2 Acceptance Summary\n\n")
    fh.write("**Headroom Gate:** PASS (EM_B0 = {:.3f} < {}).\\n\n".format(
        em_b0, os.environ["HEADROOM_THRESHOLD"]
    ))
    fh.write("**Retrieval (chat prompt)**\\n")
    fh.write(f"- EM_B0: {em_b0:.3f}\n")
    fh.write(f"- Non-empty rate (B0): {rate_b0:.3f}\n")
    fh.write(f"- EM_B1: {em_b1:.3f}\n")
    fh.write(f"- EM_B1 (no Hopfield): {em_b1_no:.3f}\n")
    fh.write(f"- Hopfield lift: {hopfield_delta:+.3f}\n")
    fh.write(f"- Non-empty rate (B1): {rate_b1:.3f}\n")
    fh.write("- Retrieval: " + ", ".join(
        part for part in [
            f"P@1={float(p_at_1):.3f}" if p_at_1 is not None else "",
            f"MRR={float(mrr):.3f}" if mrr is not None else "",
            f"Near-miss={float(near_miss):.3f}" if near_miss is not None else "",
            f"Collision={float(collision):.3f}" if collision is not None else "",
            f"CompletionΔ={float(completion_lift):+.3f}" if completion_lift is not None else "",
        ]
        if part
    ) + "\n")
    fh.write("\n**Memory-dependent uplift**\\n")
    fh.write(f"- EM_B0 (no episode prompt): {mem_em_b0:.3f}\n")
    fh.write(f"- Non-empty rate (B0): {mem_rate_b0:.3f}\n")
    fh.write(f"- EM_B1 (with memory tokens): {mem_em_b1:.3f}\n")
    fh.write(f"- Non-empty rate (B1): {mem_rate_b1:.3f}\n")
    fh.write(f"- EM lift: {lift:+.3f}\n")
    fh.write(f"- 95% bootstrap CI: [{ci_low:+.3f}, {ci_high:+.3f}]\n")
    fh.write(f"- Records evaluated: {len(records_b0)}\n")
    fh.write("\nSee probes summary at reports/m2-acceptance/probes_summary.txt "
             "and uplift details in reports/m2-acceptance/uplift_compare.txt.\n")
PY
