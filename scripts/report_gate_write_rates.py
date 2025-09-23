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
from typing import Any


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_metric_files(paths: Iterable[str]) -> Iterable[Path]:
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            yield from path.glob("*_metrics.json")
        else:
            yield path


def _parse_band(argument: str | None) -> tuple[float, float] | None:
    if not argument:
        return None
    tokens = argument.replace(",", " ").split()
    if len(tokens) != 2:
        raise ValueError("--target-band expects exactly two values")
    lower, upper = (float(token) for token in tokens)
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gate write rates")
    parser.add_argument("paths", nargs="+", help="Metrics JSON files or their directories")
    parser.add_argument("--out", type=Path, default=Path("reports/m3-gate-write-rate-summary.json"))
    parser.add_argument(
        "--target-band",
        type=str,
        default=None,
        help="Optional writes/1k tokens target band (e.g. '1,5') to annotate summaries",
    )
    args = parser.parse_args()

    try:
        band = _parse_band(args.target_band)
    except ValueError as exc:
        parser.error(str(exc))

    band_lower: float | None = None
    band_upper: float | None = None
    if band is not None:
        band_lower, band_upper = band

    rows = []
    first_in_band: Any | None = None
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
        prompt_tokens = _as_int(gate.get("prompt_tokens"))
        generated_tokens = _as_int(gate.get("generated_tokens"))
        if row["writes_per_1k_tokens"] is None:
            tokens_val = telemetry.get("writes_per_1k_tokens")
            row["writes_per_1k_tokens"] = (
                float(tokens_val) if isinstance(tokens_val, int | float) else None
            )
        prompt_tokens = prompt_tokens or _as_int(telemetry.get("prompt_tokens"))
        generated_tokens = generated_tokens or _as_int(telemetry.get("generated_tokens"))
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
        if row["writes_per_1k_tokens"] is None:
            total_tokens = 0
            for value in (prompt_tokens, generated_tokens):
                if value is not None:
                    total_tokens += value
            if total_tokens > 0:
                writes_val = _as_float(gate.get("writes"))
                if writes_val is None:
                    writes_val = _as_float(telemetry.get("writes"))
                if writes_val is not None:
                    row["writes_per_1k_tokens"] = writes_val / (total_tokens / 1000.0)
        row["writes_per_1k"] = row["writes_per_1k_tokens"]
        if (
            band_lower is not None
            and band_upper is not None
            and first_in_band is None
            and row["writes_per_1k_tokens"] is not None
        ):
            try:
                token_value = float(row["writes_per_1k_tokens"])
            except (TypeError, ValueError):
                token_value = None
            if token_value is not None and band_lower <= token_value <= band_upper:
                first_in_band = row["threshold"]
        rows.append(row)

    rows.sort(key=lambda record: (record["scenario"], record["threshold"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"runs": rows}
    if band_lower is not None and band_upper is not None:
        summary["target_band"] = {"lower": band_lower, "upper": band_upper}
        summary["first_tau_within_target_band"] = first_in_band
    with args.out.open("w", encoding="utf8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
