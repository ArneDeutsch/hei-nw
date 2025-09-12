#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
OUT="reports/m1-episodic-adapter"
MODEL="sshleifer/tiny-gpt2"

python -m hei_nw.eval.harness --mode B1 --scenario A -n 8 --seed 7 \
    --model "$MODEL" --outdir "$OUT"
python -m hei_nw.eval.harness --mode B1 --scenario B -n 8 --seed 7 \
    --model "$MODEL" --outdir "$OUT"
python -m hei_nw.eval.harness --mode B1 --scenario C -n 8 --seed 7 \
    --model "$MODEL" --outdir "$OUT"
python -m hei_nw.eval.harness --mode B1 --scenario D -n 8 --seed 7 \
    --model "$MODEL" --outdir "$OUT"
python -m hei_nw.eval.harness --mode B1 --scenario E -n 8 --seed 7 \
    --model "$MODEL" --outdir "$OUT" --baseline long-context
