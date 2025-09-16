#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT="${OUT:-reports/m2-retrieval-stack}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N="${N:-24}"
SEED="${SEED:-7}"

python -m hei_nw.eval.harness --mode B0 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT" \
  --qa.prompt_style chat --qa.max_new_tokens 16 --qa.stop '' --qa.answer_hint \
  --hopfield.steps 2 --hopfield.temperature 0.5

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT" \
  --qa.prompt_style chat --qa.max_new_tokens 16 --qa.stop '' --qa.answer_hint \
  --hopfield.steps 2 --hopfield.temperature 0.5

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT" --no-hopfield \
  --qa.prompt_style chat --qa.max_new_tokens 16 --qa.stop '' --qa.answer_hint \
  --hopfield.steps 2 --hopfield.temperature 0.5
