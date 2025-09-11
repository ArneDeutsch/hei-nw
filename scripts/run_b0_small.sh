#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
OUT=reports/baseline
python -m hei_nw.eval.harness --mode B0 --scenario A -n 64 --seed 7 --outdir "$OUT"
python -m hei_nw.eval.harness --mode B0 --scenario B -n 64 --seed 7 --outdir "$OUT"
python -m hei_nw.eval.harness --mode B0 --scenario C -n 64 --seed 7 --outdir "$OUT"
python -m hei_nw.eval.harness --mode B0 --scenario D -n 64 --seed 7 --outdir "$OUT"
python -m hei_nw.eval.harness --mode B0 --scenario E -n 64 --seed 7 --outdir "$OUT" --baseline long-context

