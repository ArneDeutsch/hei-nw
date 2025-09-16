#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT_ROOT="${OUT:-reports/m2-probes}"
MODEL="${MODEL:-models/tiny-gpt2}"
N="${N:-16}"
SEED="${SEED:-7}"

mkdir -p "$OUT_ROOT"

run_probe() {
  local suffix="$1"
  shift
  local outdir="$OUT_ROOT/$suffix"
  mkdir -p "$outdir"
  echo "[m2-probes] Running $suffix (outdir: $outdir)"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" \
    --qa.prompt_style chat --qa.max_new_tokens 16 --qa.answer_hint \
    --hopfield.steps 2 --hopfield.temperature 0.5 "$@"
}

run_probe "no_stop" --qa.stop ''
run_probe "with_stop" --qa.stop $'\n'
run_probe "retrieval_only" --qa.stop '' --dev.retrieval_only
run_probe "oracle_trace" --qa.stop '' --dev.oracle_trace
