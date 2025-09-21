#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

OUT="${OUT:-reports/m3-write-gate}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SCENARIO="A"
N="48"
SEED="13"
THRESHOLD="1.5"
PLOT_TITLE=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_m3_gate_calibration.sh [options]

Runs the B1 harness with gate telemetry enabled and produces calibration assets.
Artifacts are written to reports/m3-write-gate by default.

Options:
  --scenario SCENARIO   Scenario identifier (default: A)
  --n N                 Number of records to evaluate (default: 48)
  --seed SEED           Random seed (default: 13)
  --model MODEL         Model identifier (default: Qwen/Qwen2.5-1.5B-Instruct)
  --threshold TAU       Gate threshold τ (default: 1.5)
  --out DIR             Output directory (default: reports/m3-write-gate)
  --title TITLE         Custom title for the calibration plot
  --help, -h            Show this help message and exit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      SCENARIO="${2:-}"
      shift 2
      ;;
    --scenario=*)
      SCENARIO="${1#*=}"
      shift 1
      ;;
    --n)
      N="${2:-}"
      shift 2
      ;;
    --n=*)
      N="${1#*=}"
      shift 1
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift 1
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift 1
      ;;
    --threshold)
      THRESHOLD="${2:-}"
      shift 2
      ;;
    --threshold=*)
      THRESHOLD="${1#*=}"
      shift 1
      ;;
    --out)
      OUT="${2:-}"
      shift 2
      ;;
    --out=*)
      OUT="${1#*=}"
      shift 1
      ;;
    --title)
      PLOT_TITLE="${2:-}"
      shift 2
      ;;
    --title=*)
      PLOT_TITLE="${1#*=}"
      shift 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT"

metrics_base="${SCENARIO}_B1"
metrics_json="${OUT}/${metrics_base}_metrics.json"
telemetry_json="${OUT}/${SCENARIO}_gate_telemetry.json"
calibration_png="${OUT}/${SCENARIO}_gate_calibration.png"
trace_samples_json="${OUT}/${SCENARIO}_trace_samples.json"

echo "[m3] Running harness for scenario ${SCENARIO} (n=${N}, seed=${SEED})"
python -m hei_nw.eval.harness \
  --mode B1 \
  --scenario "$SCENARIO" \
  -n "$N" \
  --seed "$SEED" \
  --model "$MODEL" \
  --outdir "$OUT" \
  --gate.threshold "$THRESHOLD"

if [[ ! -f "$metrics_json" ]]; then
  echo "Expected metrics file not found: $metrics_json" >&2
  exit 1
fi

echo "[m3] Extracting gate telemetry"
export M3_METRICS_PATH="$metrics_json"
export M3_TELEMETRY_PATH="$telemetry_json"
export M3_TRACE_SAMPLES_PATH="$trace_samples_json"
python - <<'PY'
import json
import os
from pathlib import Path

metrics_path = Path(os.environ["M3_METRICS_PATH"])
telemetry_path = Path(os.environ["M3_TELEMETRY_PATH"])
trace_samples_path = Path(os.environ["M3_TRACE_SAMPLES_PATH"])

data = json.loads(metrics_path.read_text(encoding="utf8"))
gate = data.get("gate", {})
telemetry = dict(gate.get("telemetry") or {})
telemetry["scenario"] = data.get("dataset", {}).get("scenario")
telemetry["threshold"] = gate.get("threshold")
telemetry["writes"] = gate.get("writes")
telemetry["total"] = gate.get("total")
telemetry["write_rate"] = gate.get("write_rate")
telemetry["write_rate_per_1k"] = gate.get("write_rate_per_1k")
telemetry["pinned"] = gate.get("pinned")
telemetry["reward_flags"] = gate.get("reward_flags")
telemetry_path.write_text(json.dumps(telemetry, indent=2), encoding="utf8")

samples = gate.get("trace_samples")
if isinstance(samples, list) and samples:
    trace_samples_path.write_text(json.dumps({"trace_samples": samples}, indent=2), encoding="utf8")
elif trace_samples_path.exists():
    trace_samples_path.unlink()
PY
unset M3_METRICS_PATH M3_TELEMETRY_PATH M3_TRACE_SAMPLES_PATH

plot_cmd=("scripts/plot_gate_calibration.py" "$telemetry_json" "--out" "$calibration_png" "--metrics" "$metrics_json")
if [[ -n "$PLOT_TITLE" ]]; then
  plot_cmd+=("--title" "$PLOT_TITLE")
fi

echo "[m3] Rendering calibration curve"
python "${plot_cmd[@]}"

echo "[m3] Calibration assets written to $OUT"
