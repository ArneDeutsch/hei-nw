from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
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
    parser.add_argument(
        "--pins-only",
        action="store_true",
        help="Render the pins-only calibration slice when available.",
    )
    parser.add_argument(
        "--overlay-nonpins",
        action="store_true",
        help="Overlay non-pinned calibration data when plotting pins-only curves.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a JSON object")
    return data


def _resolve_title(meta: Mapping[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit

    scenario = meta.get("scenario")
    threshold = meta.get("threshold")
    n_value = meta.get("n")
    seed_value = meta.get("seed")
    model = meta.get("model")

    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and value == "":
            return False
        return True

    def _format_threshold(value: Any) -> str:
        if isinstance(value, int | float):
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{float_value:.2f}"

    def _format_int_like(value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int | float) and float(value).is_integer():
            return str(int(float(value)))
        if isinstance(value, str):
            try:
                return str(int(value))
            except ValueError:
                return value
        return str(value)

    if (
        _has_value(scenario)
        and _has_value(threshold)
        and _has_value(n_value)
        and _has_value(seed_value)
        and _has_value(model)
    ):
        scenario_text = str(scenario)
        threshold_text = _format_threshold(threshold)
        n_text = _format_int_like(n_value)
        seed_text = _format_int_like(seed_value)
        model_text = str(model)
        return (
            f"{scenario_text} — τ={threshold_text} — n={n_text}, "
            f"seed={seed_text}, model={model_text}"
        )

    if _has_value(scenario) and _has_value(threshold):
        scenario_text = str(scenario)
        threshold_text = _format_threshold(threshold)
        return f"Scenario {scenario_text} — τ={threshold_text}"

    if _has_value(scenario):
        return f"Scenario {scenario}"

    return "Gate calibration"


def _final_title(meta: Mapping[str, Any], explicit: str | None, pins_only: bool) -> str:
    """Return the rendered plot title, annotating pins-only slices."""

    title = _resolve_title(meta, explicit)
    if pins_only and explicit is None:
        return f"{title} (pins-only)"
    return title


def _calibration_buckets(section: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calibration = section.get("calibration")
    if isinstance(calibration, Sequence):
        return [bucket for bucket in calibration if isinstance(bucket, Mapping)]
    return []


def _calibration_points(
    buckets: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[float], list[int]]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    sizes: list[int] = []
    for bucket in buckets:
        lower = float(bucket.get("lower", 0.0))
        upper = float(bucket.get("upper", lower))
        mean_score = float(bucket.get("mean_score", (lower + upper) / 2.0))
        frac_pos = float(bucket.get("fraction_positive", 0.0))
        count = int(bucket.get("count", 0))
        x_vals.append(mean_score)
        y_vals.append(frac_pos)
        sizes.append(max(count, 1))
    return x_vals, y_vals, sizes


def main() -> None:
    args = _parse_args()
    telemetry_path: Path = args.telemetry
    telemetry = _load_json(telemetry_path)
    if args.overlay_nonpins and not args.pins_only:
        raise SystemExit("--overlay-nonpins requires --pins-only")

    primary_section: Mapping[str, Any] = telemetry
    primary_label = "Gate"
    if args.pins_only:
        pins_section = telemetry.get("pins_only")
        if not isinstance(pins_section, Mapping):
            raise SystemExit("Telemetry JSON does not contain pins_only metrics")
        primary_section = pins_section
        primary_label = "Pins"
    buckets = _calibration_buckets(primary_section)
    if not buckets:
        raise SystemExit("Telemetry JSON does not contain calibration buckets")

    x_vals, y_vals, sizes = _calibration_points(buckets)

    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.6", label="Ideal")
    ax.scatter(x_vals, y_vals, s=[10 + 5 * size for size in sizes], label=primary_label)

    if args.pins_only and args.overlay_nonpins:
        non_pins_section = telemetry.get("non_pins")
        if isinstance(non_pins_section, Mapping):
            overlay_buckets = _calibration_buckets(non_pins_section)
            if overlay_buckets:
                overlay_x, overlay_y, overlay_sizes = _calibration_points(overlay_buckets)
                ax.scatter(
                    overlay_x,
                    overlay_y,
                    s=[10 + 5 * size for size in overlay_sizes],
                    marker="s",
                    label="Non-pins",
                )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean gate score")
    ax.set_ylabel("Fraction positive")
    ax.set_title(_final_title(telemetry, args.title, args.pins_only))
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
