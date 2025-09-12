#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
OUT="reports/m2-retrieval-stack"
MODEL="tests/models/tiny-gpt2"

python -m hei_nw.eval.harness --mode B0 --scenario A -n 24 --seed 7 \
    --model "$MODEL" --outdir "$OUT"
python -m hei_nw.eval.harness --mode B1 --scenario A -n 24 --seed 7 \
    --model "$MODEL" --outdir "$OUT"
python -m hei_nw.eval.harness --mode B1 --scenario A -n 24 --seed 7 \
    --model "$MODEL" --outdir "$OUT" --no-hopfield
