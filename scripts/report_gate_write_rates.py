#!/usr/bin/env python3
"""Delegate to ``hei_nw.cli.report_gate_write_rates``."""

from __future__ import annotations

from hei_nw.cli import report_gate_write_rates as _impl

__doc__ = _impl.__doc__
__all__ = getattr(_impl, "__all__", [])


def __getattr__(name: str):  # pragma: no cover - delegation helper
    return getattr(_impl, name)


def main(argv: list[str] | None = None) -> int:
    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
