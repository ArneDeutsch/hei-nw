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
MODEL="${MODEL:-hei-nw/dummy-model}"
N="${N:-16}"
SEED="${SEED:-7}"

mkdir -p "$OUT_ROOT"

# Summary table state.
SUMMARY_HEADER="Probe | Variant | EM | Non-empty | Notes"
SUMMARY_LINES=()

# Common QA settings. Keep episode text by default; E2 overrides later.
default_args=(
  --qa.prompt_style chat
  --qa.stop ''
  --qa.max_new_tokens 8
  --qa.answer_hint
  --qa.template_policy plain
  --hopfield.steps 2
  --hopfield.temperature 0.5
)

gate_flag="--no-gate.use_for_writes"
if [[ "${USE_GATE_WRITES:-0}" -ne 0 ]]; then
  gate_flag="--gate.use_for_writes"
fi

append_summary() {
  local probe="$1"
  local variant="$2"
  local metrics_path="$3"
  local summary
  summary=$(PROBE="$probe" VARIANT="$variant" METRICS_PATH="$metrics_path" python - <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

probe = os.environ["PROBE"]
variant = os.environ["VARIANT"]
path = Path(os.environ["METRICS_PATH"])
if not path.is_file():
    print(f"[m2-probes] metrics missing for {probe} ({variant}): {path}")
    sys.exit(1)
data = json.loads(path.read_text(encoding="utf8"))
records = data.get("records") or []
if not records:
    print(f"[m2-probes] no records for {probe} ({variant})")
    sys.exit(1)
non_empty = sum(1 for rec in records if str(rec.get("prediction", "")).strip())
non_empty_rate = non_empty / len(records)
if non_empty_rate == 0:
    print(f"[m2-probes] empty predictions for {probe} ({variant})")
    sys.exit(1)
aggregate = data.get("aggregate") or {}
em = float(aggregate.get("em_relaxed", aggregate.get("em", 0.0)))
retrieval = data.get("retrieval") or {}
note_parts: list[str] = []
p_at_1 = retrieval.get("p_at_1")
if p_at_1 is not None:
    note_parts.append(f"P@1={float(p_at_1):.3f}")
mrr = retrieval.get("mrr")
if mrr is not None:
    note_parts.append(f"MRR={float(mrr):.3f}")
notes = ", ".join(note_parts) if note_parts else "-"
print(f"{probe} | {variant} | {em:.3f} | {non_empty_rate:.3f} | {notes}")
PY
  ) || {
    echo "$summary" >&2
    exit 1
  }
  SUMMARY_LINES+=("$summary")
}
run_probe() {
  local probe_id="$1"
  shift
  local outdir="$OUT_ROOT/$probe_id"
  mkdir -p "$outdir"
  echo "[m2-probes] Running $probe_id (outdir: $outdir)"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" \
    "$gate_flag" "${default_args[@]}" "$@"
}

run_e0() {
  run_probe "E0"
  append_summary "E0" "baseline" "$OUT_ROOT/E0/A_B1_metrics.json"
}

run_e1() {
  run_probe "E1" --dev.oracle_trace
  append_summary "E1" "oracle" "$OUT_ROOT/E1/A_B1_metrics.json"
}

run_e2() {
  run_probe "E2" --dev.retrieval_only
  append_summary "E2" "retrieval-only" "$OUT_ROOT/E2/A_B1_metrics.json"
}

run_e3() {
  local outdir="$OUT_ROOT/E3"
  mkdir -p "$outdir"
  echo "[m2-probes] Running E3 Hopfield on/off pair (outdir: $outdir)"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" \
    "${default_args[@]}"
  append_summary "E3" "hopfield" "$outdir/A_B1_metrics.json"
  python -m hei_nw.eval.harness --mode B1 --scenario A -n "$N" --seed "$SEED" \
    --model "$MODEL" --outdir "$outdir" --no-hopfield \
    "${default_args[@]}"
  append_summary "E3" "no-hopfield" "$outdir/A_B1_no-hopfield_metrics.json"
  HOP_WITH="$outdir/A_B1_metrics.json" HOP_WITHOUT="$outdir/A_B1_no-hopfield_metrics.json" python - <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

with_hp = json.loads(Path(os.environ["HOP_WITH"]).read_text(encoding="utf8"))
no_hp = json.loads(Path(os.environ["HOP_WITHOUT"]).read_text(encoding="utf8"))

hop_retrieval = with_hp.get("retrieval") or {}
nohop_retrieval = no_hp.get("retrieval") or {}

hop_p1 = hop_retrieval.get("p_at_1")
nohop_p1 = nohop_retrieval.get("p_at_1")
hop_completion = hop_retrieval.get("completion_lift")

def _fmt(value: float | None) -> str:
    return f"{float(value):.3f}" if value is not None else "n/a"

if hop_completion is not None and hop_completion < 0:
    print(
        f"[m2-probes] WARNING: Hopfield completion lift negative ({hop_completion:.3f})",
        file=sys.stderr,
    )
if hop_p1 is not None and nohop_p1 is not None and hop_p1 + 1e-9 < nohop_p1:
    print(
        "[m2-probes] WARNING: Hopfield P@1 ({} ) < no-Hopfield ({})".format(
            _fmt(hop_p1),
            _fmt(nohop_p1),
        ),
        file=sys.stderr,
    )
PY
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

if [[ ${#SUMMARY_LINES[@]} -gt 0 ]]; then
  echo
  echo "$SUMMARY_HEADER"
  for line in "${SUMMARY_LINES[@]}"; do
    echo "$line"
  done
fi
