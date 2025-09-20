#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

OUT="${OUT:-reports/m2-parity-guard}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N="${N:-12}"
SEED="${SEED:-7}"
THRESHOLD="${THRESHOLD:-0.1}"

mkdir -p "$OUT"

default_args=(
  --qa.prompt_style chat
  --qa.answer_hint
  --qa.max_new_tokens 16
  --qa.stop ''
  --qa.template_policy plain
)

python -m hei_nw.eval.harness --mode B0 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT" --qa.memory_dependent_baseline "${default_args[@]}"

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT" --mem.max_tokens 0 --qa.memory_dependent_baseline "${default_args[@]}"

python scripts/compare_b0_b1.py --threshold "$THRESHOLD" \
  "$OUT/A_B0_metrics.json" "$OUT/A_B1_metrics.json"
