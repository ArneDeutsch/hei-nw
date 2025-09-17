#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

OUT_ROOT="${OUT:-reports/m2-probes}"
MODEL="${MODEL:-models/tiny-gpt2}"
N="${N:-16}"
SEED="${SEED:-7}"

mkdir -p "$OUT_ROOT"

default_args=(
  --qa.prompt_style chat
  --qa.stop $'\n'
  --qa.max_new_tokens 8
  --qa.answer_hint
  --hopfield.steps 2
  --hopfield.temperature 0.5
)

run_probe() {
  local suffix="$1"
  shift
  local outdir="$OUT_ROOT/$suffix"
  mkdir -p "$outdir"
  echo "[m2-probes] Running $suffix (outdir: $outdir)"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" \
    "${default_args[@]}" "$@"
}

completion_dir="$OUT_ROOT/completion"
mkdir -p "$completion_dir"
echo "[m2-probes] Running completion pair with Hopfield on/off (dir: $completion_dir)"
python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$completion_dir" \
  "${default_args[@]}"
python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
  --model "$MODEL" --outdir "$completion_dir" --no-hopfield \
  "${default_args[@]}"

if [[ -z "${CI:-}" ]]; then
  plot="$completion_dir/completion_ablation.png"
  if [[ ! -f "$plot" ]]; then
    echo "[m2-probes] expected completion ablation plot at $plot" >&2
    exit 1
  fi
fi

run_probe "no_stop" --qa.stop ''
run_probe "retrieval_only" --qa.stop '' --dev.retrieval_only
run_probe "oracle_trace" --qa.stop '' --dev.oracle_trace
