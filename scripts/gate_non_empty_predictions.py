from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fail if the non-empty prediction rate drops below a threshold."
    )
    parser.add_argument(
        "metrics_path",
        type=Path,
        help="Path to a metrics JSON file with a 'records' list containing predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum acceptable non-empty prediction rate (default: 0.9).",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")
    return args


def compute_non_empty_rate(predictions: Sequence[str]) -> float:
    """Return the fraction of predictions that are non-empty after stripping whitespace."""
    if not predictions:
        return 0.0
    non_empty = sum(1 for prediction in predictions if prediction.strip())
    return non_empty / len(predictions)


def load_predictions(data: dict[str, Any]) -> list[str]:
    """Extract predictions from the records list in the metrics JSON."""
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("Metrics JSON must contain a 'records' list.")
    predictions: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {index} is not an object.")
        value = record.get("prediction", "")
        predictions.append("" if value is None else str(value))
    return predictions


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the non-empty prediction gate."""
    args = parse_args(argv)
    try:
        raw = args.metrics_path.read_text(encoding="utf8")
    except OSError as exc:  # pragma: no cover - exercised indirectly in subprocess tests
        print(f"Failed to read metrics file: {exc}", file=sys.stderr)
        return 2
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        print(f"Failed to parse JSON: {exc}", file=sys.stderr)
        return 2
    try:
        predictions = load_predictions(data)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rate = compute_non_empty_rate(predictions)
    print(f"Non-empty rate: {rate:.3f}")
    if rate < args.threshold:
        print(
            f"Non-empty rate {rate:.3f} below threshold {args.threshold:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
