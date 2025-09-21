#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT="${OUT:-reports/m2-retrieval-stack}"
MODEL="${MODEL:-sshleifer/tiny-gpt2}"
N="${N:-12}"
SEED="${SEED:-7}"

python -m hei_nw.eval.harness --mode B0 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT"

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT"

python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$OUT" --no-hopfield
