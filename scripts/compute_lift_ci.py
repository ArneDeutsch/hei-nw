#!/usr/bin/env python
"""Compute EM lift and bootstrap CI between two metrics JSON files."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Sequence


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b0_metrics", type=Path, help="Path to B0 metrics JSON")
    parser.add_argument("b1_metrics", type=Path, help="Path to B1 metrics JSON")
    parser.add_argument(
        "--hard-subset",
        type=Path,
        default=None,
        help="Optional file containing record indices (whitespace separated)",
    )
    parser.add_argument(
        "--resamples",
        type=_positive_int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except FileNotFoundError as exc:  # pragma: no cover - handled via tests
        raise SystemExit(f"metrics file not found: {path}") from exc


def _select_indices(total: int, subset_path: Path | None) -> list[int]:
    if subset_path is None:
        return list(range(total))
    try:
        tokens = subset_path.read_text(encoding="utf8").split()
    except FileNotFoundError as exc:  # pragma: no cover - explicit message
        raise SystemExit(f"hard subset file not found: {subset_path}") from exc
    indices = sorted({int(tok) for tok in tokens})
    if any(idx < 0 or idx >= total for idx in indices):
        raise SystemExit("hard subset indices out of range")
    return indices


def _extract_em(records: Sequence[dict], indices: Sequence[int]) -> list[float]:
    values: list[float] = []
    for idx in indices:
        record = records[idx]
        if not isinstance(record, dict):
            raise SystemExit(f"record {idx} is not an object")
        value = record.get("em_relaxed", record.get("em"))
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:  # pragma: no cover - invalid schema
            raise SystemExit(f"record {idx} lacks numeric EM value") from exc
    return values


def _non_empty_rate(records: Sequence[dict], indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    non_empty = 0
    for idx in indices:
        record = records[idx]
        prediction = record.get("prediction", "") if isinstance(record, dict) else ""
        if str(prediction).strip():
            non_empty += 1
    return non_empty / len(indices)


def _bootstrap_lift(
    em_b0: Sequence[float],
    em_b1: Sequence[float],
    *,
    resamples: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    if len(em_b0) != len(em_b1):
        raise SystemExit("B0/B1 length mismatch")
    if not em_b0:
        raise SystemExit("no records selected for lift computation")
    lift_mean = statistics.mean(em_b1) - statistics.mean(em_b0)
    boots: list[float] = []
    size = len(em_b0)
    for _ in range(resamples):
        sample = [rng.randrange(size) for _ in range(size)]
        b0_avg = statistics.mean(em_b0[i] for i in sample)
        b1_avg = statistics.mean(em_b1[i] for i in sample)
        boots.append(b1_avg - b0_avg)
    boots.sort()
    low_idx = max(0, math.floor(0.025 * (resamples - 1)))
    high_idx = min(resamples - 1, math.ceil(0.975 * (resamples - 1)))
    return lift_mean, boots[low_idx], boots[high_idx]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_b0 = _load_json(args.b0_metrics)
    metrics_b1 = _load_json(args.b1_metrics)

    records_b0 = metrics_b0.get("records", [])
    records_b1 = metrics_b1.get("records", [])
    if not isinstance(records_b0, list) or not isinstance(records_b1, list):
        raise SystemExit("metrics must contain 'records' lists")
    if len(records_b0) != len(records_b1):
        raise SystemExit("B0/B1 record counts differ")

    indices = _select_indices(len(records_b0), args.hard_subset)
    if not indices:
        raise SystemExit("no records selected for comparison")

    em_b0 = _extract_em(records_b0, indices)
    em_b1 = _extract_em(records_b1, indices)
    non_empty = _non_empty_rate(records_b1, indices)

    rng = random.Random(args.seed)
    lift_mean, ci_low, ci_high = _bootstrap_lift(em_b0, em_b1, resamples=args.resamples, rng=rng)

    print(f"Records evaluated: {len(indices)}")
    if args.hard_subset is not None:
        print(f"Hard subset: {args.hard_subset}")
    print(f"B0 EM (relaxed): {statistics.mean(em_b0):.3f}")
    print(f"B1 EM (relaxed): {statistics.mean(em_b1):.3f}")
    print(f"B1 non-empty rate: {non_empty:.3f}")
    print(f"EM lift: {lift_mean:.3f}")
    print(f"95% bootstrap CI: [{ci_low:.3f}, {ci_high:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
