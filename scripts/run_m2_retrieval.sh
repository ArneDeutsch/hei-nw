#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT="${OUT:-reports/m2-retrieval-stack}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N="${N:-24}"
SEED="${SEED:-7}"
HARD_SUBSET=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_m2_retrieval.sh [--hard-subset PATH]

Runs the headroom-aware M2 acceptance flow:
  • B0/B1/B1(no Hopfield) harness runs with chat prompts.
  • Computes the Headroom Gate from B0 relaxed EM.
  • Executes the E0–E3 isolation probes.
  • If headroom passes, invokes scripts/compare_b0_b1_m2.sh on the hard subset.
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

default_args=(
  --qa.prompt_style chat
  --qa.answer_hint
  --qa.max_new_tokens 16
  --qa.stop ''
  --hopfield.steps 2
  --hopfield.temperature 0.5
)

run_harness() {
  local mode="$1"
  shift
  echo "[m2] Running $mode (outdir: $OUT)"
  python -m hei_nw.eval.harness --mode "$mode" --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$OUT" \
    "${default_args[@]}" "$@"
}

run_harness B0
run_harness B1
run_harness B1 --no-hopfield

headroom_info=$(python - <<PY
import json
from pathlib import Path

out_path = Path("${OUT}") / "A_B0_metrics.json"
data = json.loads(out_path.read_text(encoding="utf8"))
agg = data.get("aggregate", {})
em0 = float(agg.get("em_relaxed", agg.get("em", 0.0)))
threshold = 0.7
status = "PASS" if em0 < threshold else "BLOCKED"
print(f"{status}|{em0:.3f}|{threshold:.3f}")
PY
)
IFS='|' read -r HEADROOM_STATUS HEADROOM_EM HEADROOM_THRESH <<<"$headroom_info"
if [[ "$HEADROOM_STATUS" == "PASS" ]]; then
  echo "[m2] Headroom Gate: PASS (EM_B0=${HEADROOM_EM}, threshold=${HEADROOM_THRESH})"
else
  echo "[m2] Headroom Gate: BLOCKED (EM_B0=${HEADROOM_EM} ≥ ${HEADROOM_THRESH})"
  echo "[m2] Switch to the memory-dependent baseline before claiming uplift."
fi

probe_out="${OUT}/probes"
echo "[m2] Running isolation probes (outdir: $probe_out)"
(
  OUT="$probe_out" MODEL="$MODEL" N="$N" SEED="$SEED" \
    scripts/m2_isolation_probes.sh --probe ALL
)

if [[ "$HEADROOM_STATUS" == "PASS" ]]; then
  if [[ -n "$HARD_SUBSET" ]]; then
    if [[ ! -f "$HARD_SUBSET" ]]; then
      echo "[m2] Hard subset file not found: $HARD_SUBSET" >&2
      exit 1
    fi
    echo "[m2] Running uplift compare on hard subset: $HARD_SUBSET"
    scripts/compare_b0_b1_m2.sh --hard-subset "$HARD_SUBSET"
  else
    echo "[m2] Headroom passed but no --hard-subset provided; skipping uplift compare."
  fi
else
  echo "[m2] Headroom blocked — uplift compare skipped."
fi
