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
declare -a THRESHOLD_SWEEP=()
declare -a DEFAULT_THRESHOLD_SWEEP=(0.5 1.0 1.5 2.0 2.5 3.0 3.5)
THRESHOLD_PROVIDED="false"
PIN_EVAL="false"
THRESHOLD_SWEEP_MODE="manual"
TARGET_BAND_LOWER="1"
TARGET_BAND_UPPER="5"
TARGET_BAND_PROVIDED="false"
RUN_LAST_METRICS_PATH=""
RUN_LAST_WRITES_PER_1K=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_m3_gate_calibration.sh [options]

Runs the B1 harness with gate telemetry enabled and produces calibration assets.
Artifacts are written to reports/m3-write-gate by default.

Options:
  --scenario SCENARIO       Scenario identifier (default: A)
  --n N                     Number of records to evaluate (default: 48)
  --seed SEED               Random seed (default: 13)
  --model MODEL             Model identifier (default: Qwen/Qwen2.5-1.5B-Instruct)
  --threshold TAU           Gate threshold τ (default: 1.5)
  --threshold-sweep "τ …"   Space/comma-separated list or "auto" to search τ automatically
  --target-band "LOW,HIGH"  Desired writes/1k tokens band for auto sweep (default: 1,5)
  --out DIR                 Output directory (default: reports/m3-write-gate)
  --title TITLE             Custom title for the calibration plot
  --pin-eval                Evaluate pins-only slice and emit pins-specific artifacts
  --help, -h                Show this help message and exit
USAGE
}

parse_threshold_sweep() {
  local arg="$1"
  if [[ -z "$arg" ]]; then
    echo "Missing argument for --threshold-sweep" >&2
    exit 2
  fi
  local normalized="${arg,,}"
  if [[ "$normalized" == "auto" ]]; then
    THRESHOLD_SWEEP_MODE="auto"
    THRESHOLD_SWEEP=()
    return
  fi
  THRESHOLD_SWEEP_MODE="manual"
  arg="${arg//,/ }"
  THRESHOLD_SWEEP=()
  for token in $arg; do
    if [[ -n "$token" ]]; then
      THRESHOLD_SWEEP+=("$token")
    fi
  done
  if [[ ${#THRESHOLD_SWEEP[@]} -eq 0 ]]; then
    echo "No thresholds provided for --threshold-sweep" >&2
    exit 2
  fi
}

parse_target_band() {
  local arg="$1"
  if [[ -z "$arg" ]]; then
    echo "Missing argument for --target-band" >&2
    exit 2
  fi
  arg="${arg//,/ }"
  local -a parts=()
  for token in $arg; do
    if [[ -n "$token" ]]; then
      parts+=("$token")
    fi
  done
  if [[ ${#parts[@]} -ne 2 ]]; then
    echo "--target-band expects two comma or space separated values" >&2
    exit 2
  fi
  TARGET_BAND_LOWER="${parts[0]}"
  TARGET_BAND_UPPER="${parts[1]}"
  TARGET_BAND_PROVIDED="true"
}

sanitize_tau_value() {
  local value="$1"
  value="${value// /}"
  value="${value//\//_}"
  echo "$value"
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
      THRESHOLD_PROVIDED="true"
      shift 2
      ;;
    --threshold=*)
      THRESHOLD="${1#*=}"
      THRESHOLD_PROVIDED="true"
      shift 1
      ;;
    --threshold-sweep)
      parse_threshold_sweep "${2:-}"
      shift 2
      ;;
    --threshold-sweep=*)
      parse_threshold_sweep "${1#*=}"
      shift 1
      ;;
    --target-band)
      parse_target_band "${2:-}"
      shift 2
      ;;
    --target-band=*)
      parse_target_band "${1#*=}"
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
    --pin-eval)
      PIN_EVAL="true"
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

if [[ "$THRESHOLD_SWEEP_MODE" != "auto" && ${#THRESHOLD_SWEEP[@]} -eq 0 && "$THRESHOLD_PROVIDED" != "true" ]]; then
  THRESHOLD_SWEEP=("${DEFAULT_THRESHOLD_SWEEP[@]}")
  echo "[m3] --threshold-sweep not provided; using default sweep: ${THRESHOLD_SWEEP[*]}"
fi

if [[ "$THRESHOLD_SWEEP_MODE" == "auto" ]]; then
  echo "[m3] Auto threshold sweep enabled; target band: [${TARGET_BAND_LOWER}, ${TARGET_BAND_UPPER}]"
  echo "[m3] Starting search from τ=${THRESHOLD}"
fi

if [[ "$PIN_EVAL" == "true" ]]; then
  echo "[m3] Checking for pinned records in scenario ${SCENARIO} (n=${N}, seed=${SEED})"
  if ! python - "$SCENARIO" "$N" "$SEED" <<'PY'
import sys

from hei_nw import datasets

scenario = sys.argv[1].upper()
n = int(sys.argv[2])
seed = int(sys.argv[3])

module = getattr(datasets, f"scenario_{scenario.lower()}", None)
if module is None or not hasattr(module, "generate"):
    raise SystemExit(f"Unknown scenario '{scenario}' for pin evaluation")

records = module.generate(n, seed=seed)
has_pin = any(
    isinstance(record, dict)
    and isinstance(record.get("gate_features"), dict)
    and record["gate_features"].get("pin")
    for record in records
)
if not has_pin:
    sys.stderr.write(
        f"Scenario {scenario} generated 0 pinned records (n={n}, seed={seed}).\n"
    )
    sys.exit(3)
PY
  then
    echo "[m3] Scenario ${SCENARIO} has no pinned records for the provided seed." >&2
    echo "[m3] Try a different scenario/seed or omit --pin-eval." >&2
    exit 3
  fi
fi

mkdir -p "$OUT"

metrics_base="${SCENARIO}_B1"
summary_suffix=""
if [[ "$PIN_EVAL" == "true" ]]; then
  summary_suffix="_pins"
fi

declare -a METRICS_PATHS=()
declare -a SWEEP_DIRS=()

run_calibration_for_threshold() {
  local tau="$1"
  local run_out_dir="$2"
  local plot_title="${3:-}"

  mkdir -p "$run_out_dir"

  local suffix=""
  local mode_note=""
  if [[ "$PIN_EVAL" == "true" ]]; then
    suffix="_pins"
    mode_note=" (pins-only)"
    if [[ -n "$plot_title" ]]; then
      plot_title+=" (pins-only)"
    fi
  fi

  local metrics_json="${run_out_dir}/${metrics_base}_metrics.json"
  local telemetry_json="${run_out_dir}/${SCENARIO}_gate_telemetry${suffix}.json"
  local calibration_png="${run_out_dir}/${SCENARIO}_gate_calibration${suffix}.png"
  local trace_samples_json="${run_out_dir}/${SCENARIO}_trace_samples${suffix}.json"

  echo "[m3] Running harness for scenario ${SCENARIO} at τ=${tau} (n=${N}, seed=${SEED})${mode_note}"
  local -a harness_cmd=(
    python -m hei_nw.eval.harness
    --mode B1
    --scenario "$SCENARIO"
    -n "$N"
    --seed "$SEED"
    --model "$MODEL"
    --outdir "$run_out_dir"
    --gate.threshold "$tau"
  )
  if [[ "$PIN_EVAL" == "true" ]]; then
    harness_cmd+=("--eval.pins_only")
  fi
  "${harness_cmd[@]}"

  if [[ ! -f "$metrics_json" ]]; then
    echo "Expected metrics file not found: $metrics_json" >&2
    exit 1
  fi

  echo "[m3] Extracting gate telemetry for τ=${tau}${mode_note}"
  export M3_METRICS_PATH="$metrics_json"
  export M3_TELEMETRY_PATH="$telemetry_json"
  export M3_TRACE_SAMPLES_PATH="$trace_samples_json"
  export M3_PIN_EVAL="$PIN_EVAL"
  export M3_MODEL="$MODEL"
  export M3_N="$N"
  export M3_SEED="$SEED"
  python - <<'PY'
import json
import os
from pathlib import Path

metrics_path = Path(os.environ["M3_METRICS_PATH"])
telemetry_path = Path(os.environ["M3_TELEMETRY_PATH"])
trace_samples_path = Path(os.environ["M3_TRACE_SAMPLES_PATH"])
pin_eval = os.environ.get("M3_PIN_EVAL", "false").lower() == "true"
model = os.environ.get("M3_MODEL")
n_value = os.environ.get("M3_N")
seed_value = os.environ.get("M3_SEED")


def _coerce_int(value: str | None) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value

data = json.loads(metrics_path.read_text(encoding="utf8"))
gate = data.get("gate", {})
telemetry = dict(gate.get("telemetry") or {})
telemetry["scenario"] = data.get("dataset", {}).get("scenario")
telemetry["threshold"] = gate.get("threshold")
telemetry["writes"] = gate.get("writes")
telemetry["total"] = gate.get("total")
telemetry["write_rate"] = gate.get("write_rate")
telemetry["write_rate_per_1k_tokens"] = gate.get("write_rate_per_1k_tokens")
telemetry["write_rate_per_1k_records"] = gate.get("write_rate_per_1k_records")
telemetry["write_rate_per_1k"] = telemetry.get("write_rate_per_1k_tokens")
telemetry["generated_tokens"] = gate.get("generated_tokens")
telemetry["pinned"] = gate.get("pinned")
telemetry["reward_flags"] = gate.get("reward_flags")
telemetry["pins_only_eval"] = pin_eval
if model:
    telemetry["model"] = model
n_value = _coerce_int(n_value)
if n_value is not None:
    telemetry["n"] = n_value
seed_value = _coerce_int(seed_value)
if seed_value is not None:
    telemetry["seed"] = seed_value
telemetry_path.write_text(json.dumps(telemetry, indent=2), encoding="utf8")

samples = gate.get("trace_samples")
if isinstance(samples, list) and samples:
    trace_samples_path.write_text(json.dumps({"trace_samples": samples}, indent=2), encoding="utf8")
elif trace_samples_path.exists():
    trace_samples_path.unlink()
PY
  unset M3_METRICS_PATH M3_TELEMETRY_PATH M3_TRACE_SAMPLES_PATH M3_PIN_EVAL M3_MODEL M3_N M3_SEED

  local -a plot_cmd=("scripts/plot_gate_calibration.py" "$telemetry_json" "--out" "$calibration_png" "--metrics" "$metrics_json")
  if [[ -n "$plot_title" ]]; then
    plot_cmd+=("--title" "$plot_title")
  fi
  if [[ "$PIN_EVAL" == "true" ]]; then
    plot_cmd+=("--pins-only" "--overlay-nonpins")
  fi

  echo "[m3] Rendering calibration curve for τ=${tau}${mode_note}"
  python "${plot_cmd[@]}"

  RUN_LAST_METRICS_PATH="$metrics_json"
  RUN_LAST_WRITES_PER_1K="$(python - "$metrics_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf8"))
except FileNotFoundError:
    print("")
    raise SystemExit(0)

gate = data.get("gate") or {}
telemetry = gate.get("telemetry") or {}
value = gate.get("write_rate_per_1k_tokens")
if value is None:
    value = telemetry.get("writes_per_1k_tokens")

def _to_int(raw):
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None

if value is None:
    prompt_tokens = gate.get("prompt_tokens")
    generated_tokens = gate.get("generated_tokens")
    if prompt_tokens is None:
        prompt_tokens = telemetry.get("prompt_tokens")
    if generated_tokens is None:
        generated_tokens = telemetry.get("generated_tokens")
    prompt_tokens = _to_int(prompt_tokens) or 0
    generated_tokens = _to_int(generated_tokens) or 0
    total_tokens = prompt_tokens + generated_tokens
    if total_tokens > 0:
        writes_value = gate.get("writes")
        if writes_value is None:
            writes_value = telemetry.get("writes")
        try:
            writes_value = float(writes_value)
        except (TypeError, ValueError):
            writes_value = None
        if writes_value is not None:
            value = writes_value / (total_tokens / 1000.0)

if value is None:
    print("")
else:
    print(value)
PY
)"
}

format_tau_value() {
  python - "$1" <<'PY'
import sys

try:
    value = float(sys.argv[1])
except (IndexError, ValueError):
    value = 0.0

text = f"{value:.6f}".rstrip("0").rstrip(".")
if not text:
    text = "0"
print(text)
PY
}

midpoint_tau() {
  python - "$1" "$2" <<'PY'
import sys

low = float(sys.argv[1])
high = float(sys.argv[2])
value = (low + high) / 2.0
text = f"{value:.6f}".rstrip("0").rstrip(".")
if not text:
    text = "0"
print(text)
PY
}

increase_tau_value() {
  python - "$1" <<'PY'
import sys

current = float(sys.argv[1])
if current <= 0.0:
    candidate = 0.5
else:
    candidate = current * 2.0
text = f"{candidate:.6f}".rstrip("0").rstrip(".")
if not text:
    text = "0"
print(text)
PY
}

decrease_tau_value() {
  python - "$1" <<'PY'
import sys

current = float(sys.argv[1])
if current <= 0.0:
    candidate = 0.25
else:
    candidate = current / 2.0
if candidate <= 0.0:
    candidate = 0.0001
text = f"{candidate:.6f}".rstrip("0").rstrip(".")
if not text:
    text = "0"
print(text)
PY
}

generate_sweep_summary() {
  local -n metrics_paths_ref="$1"
  local -n sweep_dirs_ref="$2"
  local -n thresholds_ref="$3"

  if [[ ${#metrics_paths_ref[@]} -eq 0 ]]; then
    echo "[m3] No metrics collected for sweep; aborting" >&2
    exit 4
  fi

  local summary_json="${OUT}/${SCENARIO}_sweep_summary${summary_suffix}.json"
  echo "[m3] Generating sweep summary at ${summary_json}"
  local -a summary_cmd=(
    python scripts/report_gate_write_rates.py
    "${metrics_paths_ref[@]}"
    --out "$summary_json"
  )
  if [[ -n "$TARGET_BAND_LOWER" && -n "$TARGET_BAND_UPPER" ]]; then
    summary_cmd+=("--target-band" "${TARGET_BAND_LOWER},${TARGET_BAND_UPPER}")
  fi
  "${summary_cmd[@]}"

  local summary_tsv="${OUT}/${SCENARIO}_sweep_summary${summary_suffix}.tsv"
  python - "$summary_tsv" "${metrics_paths_ref[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
metric_paths = [Path(p) for p in sys.argv[2:] if p]
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf8") as handle:
    handle.write(
        "scenario\ttau\twrites\twrite_rate\twrites_per_1k_tokens\twrites_per_1k_records\tpr_auc\n"
    )
    for metrics_path in metric_paths:
        data = json.loads(metrics_path.read_text(encoding="utf8"))
        dataset = data.get("dataset") or {}
        gate = data.get("gate") or {}
        telemetry = gate.get("telemetry") or {}
        scenario = dataset.get("scenario")
        threshold = gate.get("threshold")
        writes = gate.get("writes")
        if writes in (None, ""):
            writes = telemetry.get("writes")
        write_rate = gate.get("write_rate")
        writes_per_1k_tokens = gate.get("write_rate_per_1k_tokens")
        if writes_per_1k_tokens in (None, ""):
            tokens_val = telemetry.get("writes_per_1k_tokens")
            if isinstance(tokens_val, (int, float)):
                writes_per_1k_tokens = float(tokens_val)
            else:
                writes_per_1k_tokens = ""
        writes_per_1k_records = gate.get("write_rate_per_1k_records")
        if writes_per_1k_records in (None, ""):
            records_val = telemetry.get("writes_per_1k_records")
            if isinstance(records_val, (int, float)):
                writes_per_1k_records = float(records_val)
            elif write_rate not in (None, ""):
                try:
                    writes_per_1k_records = float(write_rate) * 1000.0
                except (TypeError, ValueError):
                    writes_per_1k_records = ""
            else:
                writes_per_1k_records = ""
        pr_auc = telemetry.get("pr_auc")
        row = [
            scenario,
            threshold,
            writes,
            write_rate,
            writes_per_1k_tokens,
            writes_per_1k_records,
            pr_auc,
        ]
        handle.write("\t".join("" if value is None else str(value) for value in row) + "\n")
PY

  local index_md="${OUT}/${SCENARIO}_threshold_sweep${summary_suffix}.md"
  {
    if [[ "$PIN_EVAL" == "true" ]]; then
      echo "# Threshold sweep for scenario ${SCENARIO} (pins-only)"
    else
      echo "# Threshold sweep for scenario ${SCENARIO}"
    fi
    echo ""
    echo "| τ | Calibration plot |"
    echo "| --- | --- |"
    local idx
    for ((idx = 0; idx < ${#thresholds_ref[@]}; idx++)); do
      local tau="${thresholds_ref[$idx]}"
      local dir="${sweep_dirs_ref[$idx]}"
      local plot_path="${SCENARIO}_gate_calibration"
      if [[ "$PIN_EVAL" == "true" ]]; then
        plot_path+="_pins"
      fi
      echo "| ${tau} | [Calibration plot](./${dir}/${plot_path}.png) |"
    done
  } > "$index_md"

  echo "[m3] Sweep summary written to ${summary_json} and ${summary_tsv}"
  echo "[m3] Calibration assets written to $OUT"
}

auto_threshold_sweep() {
  METRICS_PATHS=()
  SWEEP_DIRS=()
  THRESHOLD_SWEEP=()

  local tau_current
  tau_current="$(format_tau_value "$THRESHOLD")"
  local max_iterations=20
  local iteration=0
  local found_within="false"
  local tau_low=""
  local tau_high=""
  declare -A tau_seen=()

  while (( iteration < max_iterations )); do
    iteration=$((iteration + 1))
    tau_current="$(format_tau_value "$tau_current")"
    if [[ -n "${tau_seen[$tau_current]+x}" ]]; then
      echo "[m3] Auto sweep encountered repeated τ=${tau_current}; stopping search" >&2
      break
    fi
    tau_seen[$tau_current]=1

    local tau_key="$(sanitize_tau_value "$tau_current")"
    local run_dir="${OUT}/tau_${tau_key}"
    local plot_title="$PLOT_TITLE"
    if [[ -n "$plot_title" ]]; then
      plot_title+=" (τ=${tau_current})"
    fi
    run_calibration_for_threshold "$tau_current" "$run_dir" "$plot_title"

    local metrics_path="${run_dir}/${metrics_base}_metrics.json"
    local writes_per_1k="$RUN_LAST_WRITES_PER_1K"
    if [[ -z "$writes_per_1k" ]]; then
      echo "[m3] Unable to determine writes per 1k tokens for τ=${tau_current}" >&2
      exit 4
    fi

    METRICS_PATHS+=("$metrics_path")
    SWEEP_DIRS+=("tau_${tau_key}")
    THRESHOLD_SWEEP+=("$tau_current")

    local band_state
    band_state="$(python - "$writes_per_1k" "$TARGET_BAND_LOWER" "$TARGET_BAND_UPPER" <<'PY'
import sys

value = float(sys.argv[1])
lower = float(sys.argv[2])
upper = float(sys.argv[3])
if value < lower:
    print("below")
elif value > upper:
    print("above")
else:
    print("within")
PY
)"

    echo "[m3] τ=${tau_current} yields ${writes_per_1k} writes/1k tokens (${band_state})"

    if [[ "$band_state" == "within" ]]; then
      found_within="true"
      break
    elif [[ "$band_state" == "above" ]]; then
      tau_low="$tau_current"
      if [[ -n "$tau_high" ]]; then
        tau_current="$(midpoint_tau "$tau_low" "$tau_high")"
      else
        tau_current="$(increase_tau_value "$tau_current")"
      fi
    else
      tau_high="$tau_current"
      if [[ -n "$tau_low" ]]; then
        tau_current="$(midpoint_tau "$tau_low" "$tau_high")"
      else
        tau_current="$(decrease_tau_value "$tau_current")"
      fi
    fi
  done

  if [[ "$found_within" != "true" ]]; then
    echo "[m3] Auto sweep did not locate a τ with writes/1k tokens in [${TARGET_BAND_LOWER}, ${TARGET_BAND_UPPER}]" >&2
    exit 4
  fi
}

if [[ "$THRESHOLD_SWEEP_MODE" == "auto" ]]; then
  auto_threshold_sweep
  generate_sweep_summary METRICS_PATHS SWEEP_DIRS THRESHOLD_SWEEP
  exit 0
fi

if [[ ${#THRESHOLD_SWEEP[@]} -gt 0 ]]; then
  echo "[m3] Performing threshold sweep for scenario ${SCENARIO}: ${THRESHOLD_SWEEP[*]}"
  METRICS_PATHS=()
  SWEEP_DIRS=()
  for tau in "${THRESHOLD_SWEEP[@]}"; do
    tau_key="$(sanitize_tau_value "$tau")"
    run_dir="${OUT}/tau_${tau_key}"
    plot_title="$PLOT_TITLE"
    if [[ -n "$plot_title" ]]; then
      plot_title+=" (τ=${tau})"
    fi
    run_calibration_for_threshold "$tau" "$run_dir" "$plot_title"
    METRICS_PATHS+=("${run_dir}/${metrics_base}_metrics.json")
    SWEEP_DIRS+=("tau_${tau_key}")
  done

  generate_sweep_summary METRICS_PATHS SWEEP_DIRS THRESHOLD_SWEEP
  exit 0
fi

run_calibration_for_threshold "$THRESHOLD" "$OUT" "$PLOT_TITLE"

echo "[m3] Calibration assets written to $OUT"
