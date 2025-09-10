"""Common command-line interface helpers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach standard arguments to *parser* and return it."""
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device, e.g. 'cpu' or 'cuda'."
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory.")
    return parser


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse *args* using a parser with the common arguments."""
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parsed = parser.parse_args(args)
    parsed.outdir = Path(parsed.outdir)
    parsed.outdir.mkdir(parents=True, exist_ok=True)
    return parsed
