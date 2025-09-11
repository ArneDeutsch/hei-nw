#!/usr/bin/env bash
set -euo pipefail
OUT_DIR=${1:-reports/baseline}
OUT_FILE=${2:-"${OUT_DIR}/combined_report.md"}
cat "${OUT_DIR}"/*_report.md > "${OUT_FILE}"
echo "Combined report written to ${OUT_FILE}"

