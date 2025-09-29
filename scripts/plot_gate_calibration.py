#!/usr/bin/env python3
"""Thin wrapper delegating to ``hei_nw.cli.plot_gate_calibration``."""

from __future__ import annotations

from hei_nw.cli import plot_gate_calibration as _impl

__doc__ = _impl.__doc__
__all__ = getattr(_impl, "__all__", [])


def __getattr__(name: str):  # pragma: no cover - simple delegation
    return getattr(_impl, name)


def main(argv: list[str] | None = None) -> int:
    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
