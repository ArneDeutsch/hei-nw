#!/usr/bin/env python3
"""Compare B0 and B1 metrics JSON files.

Prints a small table showing EM/F1 deltas and exits with status 1
when any delta exceeds ``threshold`` (default 0.1).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _load_metrics(path: Path) -> tuple[float, float]:
    with path.open("r", encoding="utf8") as fh:
        data = json.load(fh)
    agg = data.get("aggregate", {})
    return float(agg.get("em", 0.0)), float(agg.get("f1", 0.0))


def compare_pairs(paths: Sequence[Path], threshold: float = 0.1) -> bool:
    """Compare metrics for *paths* in B0/B1 pairs.

    Parameters
    ----------
    paths:
        Sequence of JSON files provided as ``[B0, B1, B0, B1, ...]``.
    threshold:
        Maximum allowed absolute difference for EM/F1.

    Returns
    -------
    bool
        ``True`` if all pairs are within ``threshold``.
    """
    if len(paths) % 2:
        msg = "expected an even number of paths (B0/B1 pairs)"
        raise SystemExit(msg)
    print("| scenario | EM_B0 | EM_B1 | ΔEM | F1_B0 | F1_B1 | ΔF1 |")
    print("| -------- | ----- | ----- | --- | ----- | ----- | --- |")
    ok = True
    for b0, b1 in zip(paths[0::2], paths[1::2], strict=False):
        em0, f10 = _load_metrics(b0)
        em1, f11 = _load_metrics(b1)
        scenario = b0.name.split("_")[0]
        print(
            f"| {scenario} | {em0:.3f} | {em1:.3f} | {em1 - em0:.3f} | "
            f"{f10:.3f} | {f11:.3f} | {f11 - f10:.3f} |"
        )
        if abs(em1 - em0) > threshold or abs(f11 - f10) > threshold:
            ok = False
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="metrics JSON files in B0/B1 pairs")
    parser.add_argument("--threshold", type=float, default=0.1, help="maximum allowed delta")
    args = parser.parse_args(argv)
    return 0 if compare_pairs(args.paths, args.threshold) else 1


if __name__ == "__main__":
    sys.exit(main())
