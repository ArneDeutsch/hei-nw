#!/usr/bin/env bash
set -euo pipefail
metrics_path=${1:-reports/m2-retrieval-stack/A_B1_metrics.json}
python scripts/gate_non_empty_predictions.py "${metrics_path}"
