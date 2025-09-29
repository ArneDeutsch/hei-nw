#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

OUT="${OUT:-reports/m2-uplift-headroom}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N="${N:-24}"
SEED="${SEED:-7}"

default_args=(
  --qa.prompt_style chat
  --qa.answer_hint
  --qa.max_new_tokens 16
  --qa.stop ''
  --hopfield.steps 2
  --hopfield.temperature 0.5
)

gate_flag="--no-gate.use_for_writes"
if [[ "${USE_GATE_WRITES:-0}" -ne 0 ]]; then
  gate_flag="--gate.use_for_writes"
fi

run_harness() {
  local mode="$1"
  shift
  echo "[uplift] Running $mode (outdir: $OUT)"
  python -m hei_nw.eval.harness --mode "$mode" --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$OUT" \
    "$gate_flag" "${default_args[@]}" "$@"
}

run_harness B0 --qa.memory_dependent_baseline
run_harness B1 --qa.memory_dependent_baseline --qa.template_policy plain
