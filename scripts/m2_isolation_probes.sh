#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

PROBE="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe)
      PROBE="${2:-}"
      shift 2
      ;;
    --probe=*)
      PROBE="${1#*=}"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

OUT_ROOT="${OUT:-reports/m2-probes}"
MODEL="${MODEL:-models/tiny-gpt2}"
N="${N:-16}"
SEED="${SEED:-7}"

mkdir -p "$OUT_ROOT"

default_args=(
  --qa.prompt_style chat
  --qa.stop ''
  --qa.max_new_tokens 8
  --qa.answer_hint
  --hopfield.steps 2
  --hopfield.temperature 0.5
)

run_probe() {
  local probe_id="$1"
  shift
  local outdir="$OUT_ROOT/$probe_id"
  mkdir -p "$outdir"
  echo "[m2-probes] Running $probe_id (outdir: $outdir)"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" \
    "${default_args[@]}" "$@"
}

run_e0() {
  run_probe "E0"
}

run_e1() {
  run_probe "E1" --dev.oracle_trace
}

run_e2() {
  run_probe "E2" --dev.retrieval_only
}

run_e3() {
  local outdir="$OUT_ROOT/E3"
  mkdir -p "$outdir"
  echo "[m2-probes] Running E3 Hopfield on/off pair (outdir: $outdir)"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" \
    "${default_args[@]}"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" --no-hopfield \
    "${default_args[@]}"
  if [[ -z "${CI:-}" ]]; then
    local plot="$outdir/completion_ablation.png"
    if [[ ! -f "$plot" ]]; then
      echo "[m2-probes] expected completion ablation plot at $plot" >&2
      exit 1
    fi
  fi
}

case "${PROBE^^}" in
  ALL)
    run_e0
    run_e1
    run_e2
    run_e3
    ;;
  E0)
    run_e0
    ;;
  E1)
    run_e1
    ;;
  E2)
    run_e2
    ;;
  E3)
    run_e3
    ;;
  *)
    echo "Unknown probe selection: $PROBE" >&2
    exit 2
    ;;
esac
