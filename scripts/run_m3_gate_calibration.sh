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
TARGET_PER="tokens"
TARGET_PER_SOURCE="default"
TARGET_RATE_PER_1K="tokens"
TARGET_RATE_PER_SOURCE="default"
TARGET_VALUE=""
TARGET_VALUE_SET="false"
TARGET_TOLERANCE=""
TARGET_TOLERANCE_SET="false"
RUN_LAST_METRICS_PATH=""
RUN_LAST_WRITES_PER_1K_TOKENS=""
RUN_LAST_WRITES_PER_1K_RECORDS=""
RUN_LAST_TARGET_METRIC_VALUE=""
RUN_LAST_TARGET_METRIC_NAME=""
RUN_LAST_WRITE_RATE=""
AUTO_SELECTED_TAU=""
AUTO_SELECTED_METRIC_VALUE=""
AUTO_SELECTED_METRIC_NAME=""
AUTO_SELECTED_DIFF=""
AUTO_SELECTED_METRICS_PATH=""

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
  --target-per {tokens|records}
                            Metric used for the target band evaluation (default: tokens)
  --target-rate-per-1k {tokens|records}
                            Metric used for the auto target evaluation (default: tokens)
  --target VALUE           Desired writes per 1k metric value for auto τ selection
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

parse_target_per() {
  local arg="$1"
  if [[ -z "$arg" ]]; then
    echo "Missing argument for --target-per" >&2
    exit 2
  fi
  local normalized="${arg,,}"
  case "$normalized" in
    token|tokens)
      set_target_metric_kind "tokens" "target-per"
      ;;
    record|records)
      set_target_metric_kind "records" "target-per"
      ;;
    *)
      echo "Invalid value for --target-per: $arg" >&2
      exit 2
      ;;
  esac
}

set_target_metric_kind() {
  local value="$1"
  local source="$2"
  if [[ "$TARGET_PER_SOURCE" != "default" && "$TARGET_PER" != "$value" ]]; then
    echo "Conflicting target metric specifications: --${source} vs previous setting" >&2
    exit 2
  fi
  if [[ "$TARGET_RATE_PER_SOURCE" != "default" && "$TARGET_RATE_PER_1K" != "$value" ]]; then
    echo "Conflicting target metric specifications: --${source} vs previous setting" >&2
    exit 2
  fi
  TARGET_PER="$value"
  TARGET_PER_SOURCE="$source"
  TARGET_RATE_PER_1K="$value"
  TARGET_RATE_PER_SOURCE="$source"
}

parse_target_rate_per_1k() {
  local arg="$1"
  if [[ -z "$arg" ]]; then
    echo "Missing argument for --target-rate-per-1k" >&2
    exit 2
  fi
  local normalized="${arg,,}"
  case "$normalized" in
    token|tokens)
      set_target_metric_kind "tokens" "target-rate-per-1k"
      ;;
    record|records)
      set_target_metric_kind "records" "target-rate-per-1k"
      ;;
    *)
      echo "Invalid value for --target-rate-per-1k: $arg" >&2
      exit 2
      ;;
  esac
}

parse_target_value() {
  local arg="$1"
  if [[ -z "$arg" ]]; then
    echo "Missing argument for --target" >&2
    exit 2
  fi
  if ! python - "$arg" <<'PY' >/dev/null 2>&1; then
import sys
try:
    float(sys.argv[1])
except (IndexError, ValueError):
    raise SystemExit(1)
PY
    echo "--target expects a numeric value" >&2
    exit 2
  fi
  TARGET_VALUE="$arg"
  TARGET_VALUE_SET="true"
}

sanitize_tau_value() {
  local value="$1"
  value="${value// /}"
  value="${value//\//_}"
  echo "$value"
}

compute_target_tolerance() {
  if [[ "$TARGET_VALUE_SET" != "true" ]]; then
    TARGET_TOLERANCE=""
    TARGET_TOLERANCE_SET="false"
    return
  fi

  if [[ "$TARGET_BAND_PROVIDED" == "true" ]]; then
    local band_tol
    if band_tol="$(
      python - "$TARGET_VALUE" "$TARGET_BAND_LOWER" "$TARGET_BAND_UPPER" <<'PY'
import math
import sys

target = float(sys.argv[1])
lower = float(sys.argv[2])
upper = float(sys.argv[3])
tolerance = max(abs(target - lower), abs(upper - target))
print(f"{tolerance:.6f}")
PY
    )"; then
        TARGET_TOLERANCE="$band_tol"
        TARGET_TOLERANCE_SET="true"
        return
      fi
  fi

  local default_tol
  default_tol="$(
    python - "$TARGET_VALUE" <<'PY'
import math
import sys

target = abs(float(sys.argv[1]))
base = max(target, 1.0)
tolerance = max(0.1, 0.05 * base)
print(f"{tolerance:.6f}")
PY
  )"
  TARGET_TOLERANCE="$default_tol"
  TARGET_TOLERANCE_SET="true"
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
    --target-per)
      parse_target_per "${2:-}"
      shift 2
      ;;
    --target-per=*)
      parse_target_per "${1#*=}"
      shift 1
      ;;
    --target-rate-per-1k)
      parse_target_rate_per_1k "${2:-}"
      shift 2
      ;;
    --target-rate-per-1k=*)
      parse_target_rate_per_1k "${1#*=}"
      shift 1
      ;;
    --target)
      parse_target_value "${2:-}"
      shift 2
      ;;
    --target=*)
      parse_target_value "${1#*=}"
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

compute_target_tolerance

if [[ "$THRESHOLD_SWEEP_MODE" == "auto" ]]; then
  if [[ "$TARGET_VALUE_SET" == "true" ]]; then
    if [[ "$TARGET_TOLERANCE_SET" == "true" && -n "$TARGET_TOLERANCE" ]]; then
      echo "[m3] Auto threshold sweep targeting ${TARGET_VALUE} writes/1k ${TARGET_PER} (tolerance ±${TARGET_TOLERANCE})"
    else
      echo "[m3] Auto threshold sweep targeting ${TARGET_VALUE} writes/1k ${TARGET_PER}"
    fi
  else
    echo "[m3] Auto threshold sweep enabled; target band: [${TARGET_BAND_LOWER}, ${TARGET_BAND_UPPER}] per 1k ${TARGET_PER}"
  fi
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
    --no-gate.allow_label_fallback
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
  mapfile -t _metric_lines < <(python - "$metrics_json" <<'PY'
import json
import sys
from pathlib import Path


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf8"))
gate = data.get("gate") or {}
telemetry = gate.get("telemetry") or {}

tokens_value = gate.get("write_rate_per_1k_tokens")
if tokens_value is None:
    candidate = telemetry.get("writes_per_1k_tokens")
    if isinstance(candidate, (int, float)):
        tokens_value = float(candidate)
    else:
        writes_val = _as_float(gate.get("writes"))
        if writes_val is None:
            writes_val = _as_float(telemetry.get("writes"))
        prompt_tokens = _as_int(gate.get("prompt_tokens"))
        if prompt_tokens is None:
            prompt_tokens = _as_int(telemetry.get("prompt_tokens"))
        generated_tokens = _as_int(gate.get("generated_tokens"))
        if generated_tokens is None:
            generated_tokens = _as_int(telemetry.get("generated_tokens"))
        total_tokens = 0
        for component in (prompt_tokens, generated_tokens):
            if component is not None:
                total_tokens += component
        if total_tokens > 0 and writes_val is not None:
            tokens_value = writes_val / (total_tokens / 1000.0)

records_value = gate.get("write_rate_per_1k_records")
if records_value is None:
    candidate = telemetry.get("writes_per_1k_records")
    if isinstance(candidate, (int, float)):
        records_value = float(candidate)
    else:
        write_rate = _as_float(gate.get("write_rate"))
        if write_rate is None:
            write_rate = _as_float(telemetry.get("write_rate"))
        if write_rate is not None:
            records_value = write_rate * 1000.0

if tokens_value is None:
    print("")
else:
    print(tokens_value)

if records_value is None:
    print("")
else:
    print(records_value)

write_rate_value = gate.get("write_rate")
if write_rate_value is None:
    write_rate_value = telemetry.get("write_rate")
if write_rate_value is None:
    print("")
else:
    print(write_rate_value)
PY
)
  if [[ ${#_metric_lines[@]} -ge 1 ]]; then
    RUN_LAST_WRITES_PER_1K_TOKENS="${_metric_lines[0]}"
  else
    RUN_LAST_WRITES_PER_1K_TOKENS=""
  fi
  if [[ ${#_metric_lines[@]} -ge 2 ]]; then
    RUN_LAST_WRITES_PER_1K_RECORDS="${_metric_lines[1]}"
  else
    RUN_LAST_WRITES_PER_1K_RECORDS=""
  fi
  if [[ ${#_metric_lines[@]} -ge 3 ]]; then
    RUN_LAST_WRITE_RATE="${_metric_lines[2]}"
  else
    RUN_LAST_WRITE_RATE=""
  fi
  if [[ "$TARGET_PER" == "records" ]]; then
    RUN_LAST_TARGET_METRIC_VALUE="$RUN_LAST_WRITES_PER_1K_RECORDS"
    RUN_LAST_TARGET_METRIC_NAME="writes_per_1k_records"
  else
    RUN_LAST_TARGET_METRIC_VALUE="$RUN_LAST_WRITES_PER_1K_TOKENS"
    RUN_LAST_TARGET_METRIC_NAME="writes_per_1k_tokens"
  fi
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
  if [[ "$TARGET_VALUE_SET" == "true" ]]; then
    summary_cmd+=("--target-value" "$TARGET_VALUE")
  fi
  summary_cmd+=("--target-per" "$TARGET_PER")
  if [[ -n "$AUTO_SELECTED_TAU" ]]; then
    summary_cmd+=("--auto-selected-tau" "$AUTO_SELECTED_TAU")
    if [[ -n "$AUTO_SELECTED_METRIC_NAME" ]]; then
      summary_cmd+=("--auto-selected-metric" "$AUTO_SELECTED_METRIC_NAME")
    fi
    if [[ -n "$AUTO_SELECTED_METRIC_VALUE" ]]; then
      summary_cmd+=("--auto-selected-metric-value" "$AUTO_SELECTED_METRIC_VALUE")
    fi
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

persist_auto_selected_tau() {
  local tau_value="$1"
  local metric_name="$2"
  local metric_value="$3"
  local target_metric="$4"
  local target_value="$5"
  local tolerance_value="$6"
  local band_lower="$7"
  local band_upper="$8"
  local metrics_path="$9"
  local diff_value="${10}"
  local out_path="${OUT}/${SCENARIO}_auto_selected_tau${summary_suffix}.json"
  python - "$out_path" "$SCENARIO" "$PIN_EVAL" "$tau_value" "$metric_name" "$metric_value" \
    "$target_metric" "$target_value" "$tolerance_value" "$band_lower" "$band_upper" \
    "$metrics_path" "$diff_value" <<'PY'
import json
import math
import sys
from pathlib import Path


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


out_path = Path(sys.argv[1])
scenario = sys.argv[2]
pins_flag = sys.argv[3].lower() == "true"
tau_raw = sys.argv[4]
metric_name = sys.argv[5] or None
metric_value_raw = sys.argv[6]
target_metric = sys.argv[7] or None
target_value_raw = sys.argv[8]
tolerance_raw = sys.argv[9]
band_lower_raw = sys.argv[10]
band_upper_raw = sys.argv[11]
metrics_path_raw = sys.argv[12]
diff_raw = sys.argv[13] if len(sys.argv) > 13 else None

data: dict[str, object] = {
    "scenario": scenario,
    "tau": _as_float(tau_raw),
    "metric": metric_name,
    "metric_value": _as_float(metric_value_raw),
}

if pins_flag:
    data["pins_only"] = True
if target_metric:
    data["target_metric"] = target_metric
target_value = _as_float(target_value_raw)
if target_value is not None:
    data["target_value"] = target_value
    if target_metric:
        data.setdefault("target_metric", target_metric)
tolerance = _as_float(tolerance_raw)
if tolerance is not None:
    data["tolerance"] = tolerance
band_lower = _as_float(band_lower_raw)
band_upper = _as_float(band_upper_raw)
if band_lower is not None or band_upper is not None:
    data["target_band"] = {
        "lower": band_lower,
        "upper": band_upper,
    }
diff_value = _as_float(diff_raw)
if diff_value is not None:
    data["delta_from_target"] = diff_value

metrics_path = Path(metrics_path_raw) if metrics_path_raw else None
if metrics_path and metrics_path.exists():
    data["metrics_path"] = str(metrics_path)
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf8"))
    except json.JSONDecodeError:
        metrics = None
    if isinstance(metrics, dict):
        gate = metrics.get("gate") or {}
        telemetry = gate.get("telemetry") or {}
        write_rate = gate.get("write_rate")
        if write_rate is None:
            write_rate = telemetry.get("write_rate")
        write_rate_value = _as_float(write_rate)
        if write_rate_value is not None:
            data["write_rate"] = write_rate_value
        writes_per_1k_tokens = gate.get("write_rate_per_1k_tokens")
        if writes_per_1k_tokens is None:
            writes_per_1k_tokens = telemetry.get("writes_per_1k_tokens")
        tokens_value = _as_float(writes_per_1k_tokens)
        if tokens_value is not None:
            data.setdefault("writes_per_1k_tokens", tokens_value)
        writes_per_1k_records = gate.get("write_rate_per_1k_records")
        if writes_per_1k_records is None:
            writes_per_1k_records = telemetry.get("writes_per_1k_records")
        records_value = _as_float(writes_per_1k_records)
        if records_value is not None:
            data.setdefault("writes_per_1k_records", records_value)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(data, indent=2), encoding="utf8")
PY
  echo "[m3] Auto-selected τ metadata written to ${out_path}"
}

auto_threshold_sweep() {
  METRICS_PATHS=()
  SWEEP_DIRS=()
  THRESHOLD_SWEEP=()
  AUTO_SELECTED_TAU=""
  AUTO_SELECTED_METRIC_VALUE=""
  AUTO_SELECTED_METRIC_NAME=""
  AUTO_SELECTED_DIFF=""
  AUTO_SELECTED_METRICS_PATH=""

  local tau_current
  tau_current="$(format_tau_value "$THRESHOLD")"
  local max_iterations=20
  local iteration=0
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
    local metric_value="$RUN_LAST_TARGET_METRIC_VALUE"
    local metric_name="$RUN_LAST_TARGET_METRIC_NAME"
    if [[ -z "$metric_value" ]]; then
      echo "[m3] Unable to determine writes per 1k ${TARGET_PER} for τ=${tau_current}" >&2
      exit 4
    fi
    if [[ -z "$metric_name" ]]; then
      if [[ "$TARGET_PER" == "records" ]]; then
        metric_name="writes_per_1k_records"
      else
        metric_name="writes_per_1k_tokens"
      fi
    fi

    METRICS_PATHS+=("$metrics_path")
    SWEEP_DIRS+=("tau_${tau_key}")
    THRESHOLD_SWEEP+=("$tau_current")

    local band_state="unknown"
    local auto_direction=""
    local auto_diff=""
    if [[ "$TARGET_VALUE_SET" == "true" ]]; then
      local eval_output
      eval_output="$(
        python - "$metric_value" "$TARGET_VALUE" "$TARGET_TOLERANCE" <<'PY'
import math
import sys

value = float(sys.argv[1])
target = float(sys.argv[2])
tolerance = None
if len(sys.argv) >= 4 and sys.argv[3]:
    tolerance = float(sys.argv[3])
delta = value - target
if delta > 0:
    direction = "above"
elif delta < 0:
    direction = "below"
else:
    direction = "within"
if tolerance is not None and math.isfinite(tolerance) and abs(delta) <= tolerance:
    direction = "within"
diff_text = f"{abs(delta):.6f}".rstrip("0").rstrip(".")
if not diff_text:
    diff_text = "0"
print(f"AUTO_DIRECTION={direction}")
print(f"AUTO_DIFF={diff_text}")
PY
      )"
      eval "$eval_output"
      auto_direction="${AUTO_DIRECTION:-}"
      auto_diff="${AUTO_DIFF:-}"
      unset AUTO_DIRECTION AUTO_DIFF
      band_state="$auto_direction"
    else
      band_state="$(
        python - "$metric_value" "$TARGET_BAND_LOWER" "$TARGET_BAND_UPPER" <<'PY' 2>/dev/null
import sys

try:
    value = float(sys.argv[1])
    lower = float(sys.argv[2])
    upper = float(sys.argv[3])
except (IndexError, ValueError):
    print("invalid")
    raise SystemExit(0)

if value < lower:
    print("below")
elif value > upper:
    print("above")
else:
    print("within")
PY
      )"
    fi

    local metric_label="tokens"
    if [[ "$TARGET_PER" == "records" ]]; then
      metric_label="records"
    fi

    local log_message="[m3] τ=${tau_current} yields ${metric_value} writes/1k ${metric_label}"
    if [[ "$TARGET_VALUE_SET" == "true" ]]; then
      if [[ -n "$auto_diff" ]]; then
        log_message+=" (Δ=${auto_diff}"
        case "$auto_direction" in
          above)
            log_message+="; above target"
            ;;
          below)
            log_message+="; below target"
            ;;
          within)
            log_message+="; within tolerance"
            ;;
        esac
        log_message+=")"
      elif [[ -n "$auto_direction" ]]; then
        log_message+=" (${auto_direction} target)"
      fi
    else
      log_message+=" (${band_state})"
    fi
    echo "$log_message"

    if [[ "$TARGET_VALUE_SET" != "true" && "$band_state" == "invalid" ]]; then
      echo "[m3] Invalid metric value '${metric_value}' for τ=${tau_current}" >&2
      exit 4
    fi

    local should_update="false"
    if [[ "$TARGET_VALUE_SET" == "true" ]]; then
      if [[ -z "$AUTO_SELECTED_TAU" || -z "$AUTO_SELECTED_DIFF" ]]; then
        should_update="true"
      else
        local compare
        compare="$(
          python - "$auto_diff" "$AUTO_SELECTED_DIFF" "$tau_current" "$AUTO_SELECTED_TAU" <<'PY'
import math
import sys

new_diff = float(sys.argv[1])
prev_diff = float(sys.argv[2])
new_tau = float(sys.argv[3])
prev_tau = float(sys.argv[4])
if new_diff < prev_diff:
    print("update")
elif math.isclose(new_diff, prev_diff, rel_tol=1e-9, abs_tol=1e-9) and new_tau < prev_tau:
    print("update")
else:
    print("keep")
PY
        )"
        if [[ "$compare" == "update" ]]; then
          should_update="true"
        fi
      fi
    elif [[ "$band_state" == "within" ]]; then
      local update_decision
      update_decision="$(
        python - "$AUTO_SELECTED_TAU" "$tau_current" <<'PY'
import sys

previous = sys.argv[1]
current = sys.argv[2]
if not previous:
    print("update")
else:
    try:
        prev_val = float(previous)
        curr_val = float(current)
    except ValueError:
        print("keep")
    else:
        if curr_val < prev_val:
            print("update")
        else:
            print("keep")
PY
      )"
      if [[ "$update_decision" == "update" || -z "$AUTO_SELECTED_TAU" ]]; then
        should_update="true"
      fi
    fi

    if [[ "$should_update" == "true" ]]; then
      AUTO_SELECTED_TAU="$tau_current"
      AUTO_SELECTED_METRIC_VALUE="$metric_value"
      AUTO_SELECTED_METRIC_NAME="$metric_name"
      AUTO_SELECTED_DIFF="$auto_diff"
      AUTO_SELECTED_METRICS_PATH="$metrics_path"
    fi

    if [[ "$band_state" == "within" ]]; then
      tau_high="$tau_current"
      if [[ -n "$tau_low" ]]; then
        tau_current="$(midpoint_tau "$tau_low" "$tau_high")"
      else
        tau_current="$(decrease_tau_value "$tau_current")"
      fi
      continue
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

  if [[ -z "$AUTO_SELECTED_TAU" ]]; then
    if [[ "$TARGET_VALUE_SET" == "true" ]]; then
      echo "[m3] Auto sweep failed to select τ for target ${TARGET_VALUE} writes/1k ${TARGET_PER}" >&2
    else
      echo "[m3] Auto sweep did not locate a τ with writes/1k ${TARGET_PER} in [${TARGET_BAND_LOWER}, ${TARGET_BAND_UPPER}]" >&2
    fi
    exit 4
  fi

  if [[ "$TARGET_VALUE_SET" == "true" && -n "$AUTO_SELECTED_DIFF" ]]; then
    echo "[m3] Auto-selected τ=${AUTO_SELECTED_TAU} with ${AUTO_SELECTED_METRIC_VALUE} writes/1k ${TARGET_PER} (Δ=${AUTO_SELECTED_DIFF})"
  else
    echo "[m3] Auto-selected τ=${AUTO_SELECTED_TAU} with ${AUTO_SELECTED_METRIC_VALUE} writes/1k ${TARGET_PER}"
  fi
}

if [[ "$THRESHOLD_SWEEP_MODE" == "auto" ]]; then
  auto_threshold_sweep
  persist_auto_selected_tau \
    "$AUTO_SELECTED_TAU" \
    "$AUTO_SELECTED_METRIC_NAME" \
    "$AUTO_SELECTED_METRIC_VALUE" \
    "$TARGET_PER" \
    "$TARGET_VALUE" \
    "$TARGET_TOLERANCE" \
    "$TARGET_BAND_LOWER" \
    "$TARGET_BAND_UPPER" \
    "$AUTO_SELECTED_METRICS_PATH" \
    "$AUTO_SELECTED_DIFF"
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
