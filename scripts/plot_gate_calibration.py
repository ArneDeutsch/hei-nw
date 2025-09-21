#!/usr/bin/env python3
"""Render a calibration plot for neuromodulated gate telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot gate calibration curve from telemetry JSON",
    )
    parser.add_argument(
        "telemetry",
        type=Path,
        help="Path to gate telemetry JSON produced by the evaluation harness",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/m3-write-gate/gate_calibration.png"),
        help="Destination path for the calibration PNG",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional metrics JSON to update with the calibration plot path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title. Defaults to scenario/threshold metadata when available.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a JSON object")
    return data


def _resolve_title(meta: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    scenario = meta.get("scenario")
    threshold = meta.get("threshold")
    if scenario and isinstance(threshold, (int, float)):
        return f"Scenario {scenario} — τ={threshold:.2f}"
    if scenario:
        return f"Scenario {scenario}"
    return "Gate calibration"


def main() -> None:
    args = _parse_args()
    telemetry_path: Path = args.telemetry
    telemetry = _load_json(telemetry_path)
    calibration = telemetry.get("calibration")
    if not calibration:
        raise SystemExit("Telemetry JSON does not contain calibration buckets")
    buckets: list[dict[str, Any]] = []
    for bucket in calibration:
        if isinstance(bucket, dict):
            buckets.append(bucket)
    if not buckets:
        raise SystemExit("No valid calibration buckets found")

    x_vals = []
    y_vals = []
    sizes = []
    for bucket in buckets:
        lower = float(bucket.get("lower", 0.0))
        upper = float(bucket.get("upper", lower))
        mean_score = float(bucket.get("mean_score", (lower + upper) / 2.0))
        frac_pos = float(bucket.get("fraction_positive", 0.0))
        count = int(bucket.get("count", 0))
        x_vals.append(mean_score)
        y_vals.append(frac_pos)
        sizes.append(max(count, 1))

    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.6", label="Ideal")
    ax.scatter(x_vals, y_vals, c="tab:blue", s=[10 + 5 * size for size in sizes], label="Gate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean gate score")
    ax.set_ylabel("Fraction positive")
    ax.set_title(_resolve_title(telemetry, args.title))
    ax.legend(loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)

    if args.metrics:
        metrics_path: Path = args.metrics
        metrics = _load_json(metrics_path)
        gate_section = metrics.setdefault("gate", {})
        if isinstance(gate_section, dict):
            gate_section["calibration_plot"] = str(args.out)
            gate_section.setdefault("telemetry_path", str(telemetry_path))
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
