from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

DEFAULT_INVALID_PREFIXES = ("<", "•")
DEFAULT_INVALID_CASELESS_PREFIXES = ("human:", "user:", "assistant:")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fail if predictions are empty or begin with disallowed/non-alphabetic tokens."
        )
    )
    parser.add_argument(
        "metrics_path",
        type=Path,
        help="Path to a metrics JSON file with a 'records' list containing predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum acceptable non-empty prediction rate (default: 1.0).",
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


def first_token(text: str) -> str:
    stripped = text.lstrip()
    if not stripped:
        return ""
    return stripped.split(maxsplit=1)[0]


def find_invalid_first_tokens(
    predictions: Sequence[str],
    *,
    prefixes: Sequence[str] = DEFAULT_INVALID_PREFIXES,
    caseless_prefixes: Sequence[str] = DEFAULT_INVALID_CASELESS_PREFIXES,
) -> list[tuple[int, str, str]]:
    """Return indices, tokens, and reasons for disallowed first-token patterns."""
    invalid: list[tuple[int, str, str]] = []
    prefix_tuple = tuple(prefixes)
    caseless_tuple = tuple(caseless_prefixes)
    for index, prediction in enumerate(predictions):
        token = first_token(prediction)
        if not token:
            continue
        lower_token = token.lower()
        if prefix_tuple and token.startswith(prefix_tuple):
            invalid.append((index, token, "disallowed prefix"))
            continue
        if caseless_tuple and lower_token.startswith(caseless_tuple):
            invalid.append((index, token, "disallowed prefix"))
            continue
        if not token.isalpha():
            invalid.append((index, token, "non-alphabetic first token"))
    return invalid


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

    invalid_tokens = find_invalid_first_tokens(predictions)
    if invalid_tokens:
        indices = ", ".join(f"{idx}:{tok} ({reason})" for idx, tok, reason in invalid_tokens)
        print(
            f"Invalid first tokens detected at {indices}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
