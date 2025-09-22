#!/usr/bin/env python3
"""Summarize write-gate metrics from harness outputs.

Usage:
    python scripts/report_gate_write_rates.py reports/*/A_B1_metrics.json --out reports/summary.json

The script extracts write rate, counts, and threshold settings from each
metrics file and emits a JSON summary sorted by threshold. The summary exposes
both writes per 1k tokens (current target) and per 1k records for backward
compatibility.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path


def _iter_metric_files(paths: Iterable[str]) -> Iterable[Path]:
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            yield from path.glob("*_metrics.json")
        else:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gate write rates")
    parser.add_argument("paths", nargs="+", help="Metrics JSON files or their directories")
    parser.add_argument("--out", type=Path, default=Path("reports/m3-gate-write-rate-summary.json"))
    args = parser.parse_args()

    rows = []
    for metrics_path in _iter_metric_files(args.paths):
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf8") as handle:
            data = json.load(handle)
        gate = data.get("gate", {})
        telemetry = gate.get("telemetry") or {}
        row = {
            "metrics_path": str(metrics_path),
            "scenario": data.get("dataset", {}).get("scenario"),
            "threshold": gate.get("threshold"),
            "write_rate": gate.get("write_rate"),
            "writes": gate.get("writes"),
            "total": gate.get("total"),
            "writes_per_1k_tokens": gate.get("write_rate_per_1k_tokens"),
            "writes_per_1k_records": gate.get("write_rate_per_1k_records"),
        }
        if row["writes_per_1k_tokens"] is None:
            tokens_val = telemetry.get("writes_per_1k_tokens")
            row["writes_per_1k_tokens"] = (
                float(tokens_val) if isinstance(tokens_val, int | float) else None
            )
        if row["writes_per_1k_records"] is None:
            clutter = telemetry.get("writes_per_1k_records")
            if isinstance(clutter, int | float):
                row["writes_per_1k_records"] = float(clutter)
            else:
                try:
                    if row["write_rate"] is not None:
                        row["writes_per_1k_records"] = float(row["write_rate"]) * 1000.0
                except TypeError:
                    row["writes_per_1k_records"] = None
        row["writes_per_1k"] = row["writes_per_1k_tokens"]
        rows.append(row)

    rows.sort(key=lambda record: (record["scenario"], record["threshold"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as handle:
        json.dump({"runs": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
