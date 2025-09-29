#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

# Thin wrapper preserving --threshold-sweep, --target-rate-per-1k, --target VALUE, --pin-eval, and --eval.pins_only flags.
python -m hei_nw.cli.run_m3_gate_calibration "$@"
