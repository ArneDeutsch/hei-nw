#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-reports/m2-retrieval-stack}"
HARD_SUBSET=""
CI_SAMPLES="${CI_SAMPLES:-1000}"

usage() {
  cat <<'USAGE'
Usage: scripts/compare_b0_b1_m2.sh [--hard-subset PATH] [--outdir PATH]

Summarises B0 vs B1 metrics for M2, optionally restricting to a hard subset
of record indices and computing a paired bootstrap 95% CI on EM lift.

Environment variables:
  OUTDIR       Directory containing A_B0_metrics.json and A_B1_metrics.json.
  CI_SAMPLES   Number of bootstrap resamples (default: 1000).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hard-subset)
      HARD_SUBSET="${2:-}"
      shift 2
      ;;
    --hard-subset=*)
      HARD_SUBSET="${1#*=}"
      shift 1
      ;;
    --outdir)
      OUTDIR="${2:-}"
      shift 2
      ;;
    --outdir=*)
      OUTDIR="${1#*=}"
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

B0_PATH="${OUTDIR}/A_B0_metrics.json"
B1_PATH="${OUTDIR}/A_B1_metrics.json"
if [[ ! -f "$B0_PATH" || ! -f "$B1_PATH" ]]; then
  echo "Missing metrics JSON under $OUTDIR" >&2
  exit 1
fi

python scripts/compare_b0_b1.py "$B0_PATH" "$B1_PATH" || true

python - <<PY
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Sequence

b0_path = Path("$B0_PATH")
b1_path = Path("$B1_PATH")
subset_path = Path("$HARD_SUBSET") if "$HARD_SUBSET" else None
ci_samples = int("$CI_SAMPLES")

with b0_path.open("r", encoding="utf8") as fh:
    metrics_b0 = json.load(fh)
with b1_path.open("r", encoding="utf8") as fh:
    metrics_b1 = json.load(fh)

records_b0 = metrics_b0.get("records", [])
records_b1 = metrics_b1.get("records", [])
if len(records_b0) != len(records_b1):
    raise SystemExit("Record count mismatch between B0 and B1 metrics")

indices: Sequence[int]
if subset_path is not None:
    if not subset_path.is_file():
        raise SystemExit(f"Hard subset file not found: {subset_path}")
    tokens = subset_path.read_text(encoding="utf8").split()
    indices = sorted({int(tok) for tok in tokens})
else:
    indices = list(range(len(records_b0)))

if not indices:
    raise SystemExit("No records selected for comparison")

em_b0 = [float(records_b0[i].get("em_relaxed", records_b0[i].get("em", 0.0))) for i in indices]
em_b1 = [float(records_b1[i].get("em_relaxed", records_b1[i].get("em", 0.0))) for i in indices]
non_empty = sum(1 for i in indices if str(records_b1[i].get("prediction", "")).strip()) / len(indices)

retrieval = metrics_b1.get("retrieval", {}) or {}
p_at_1 = retrieval.get("p_at_1")
mrr = retrieval.get("mrr")

lift_mean = mean(em_b1) - mean(em_b0)

rng = random.Random(0)
boots: list[float] = []
for _ in range(ci_samples):
    sample = [rng.choice(indices) for _ in indices]
    b0_avg = mean(em_b0[i] for i in sample)
    b1_avg = mean(em_b1[i] for i in sample)
    boots.append(b1_avg - b0_avg)
boots.sort()
low_idx = max(0, math.floor(0.025 * (ci_samples - 1)))
high_idx = min(len(boots) - 1, math.ceil(0.975 * (ci_samples - 1)))
ci_low = boots[low_idx]
ci_high = boots[high_idx]

subset_label = "full set" if subset_path is None else f"hard subset ({subset_path})"
print(f"[compare] Evaluated {len(indices)} items on {subset_label}.")
print(f"[compare] B0 EM (relaxed) = {mean(em_b0):.3f}")
print(f"[compare] B1 EM (relaxed) = {mean(em_b1):.3f}")
print(f"[compare] Non-empty rate (B1) = {non_empty:.3f}")
if p_at_1 is not None or mrr is not None:
    parts: list[str] = []
    if p_at_1 is not None:
        parts.append(f"P@1={float(p_at_1):.3f}")
    if mrr is not None:
        parts.append(f"MRR={float(mrr):.3f}")
    print("[compare] Retrieval health: " + ", ".join(parts))
else:
    print("[compare] Retrieval health: unavailable")
print(
    "[compare] EM lift = {lift:.3f} (95% bootstrap CI: [{lo:.3f}, {hi:.3f}])".format(
        lift=lift_mean, lo=ci_low, hi=ci_high
    )
)
if lift_mean < 0.30 or ci_low <= 0.0:
    raise SystemExit(
        f"Lift {lift_mean:.3f} with CI [{ci_low:.3f}, {ci_high:.3f}] does not meet headroom-aware acceptance"
    )
PY
