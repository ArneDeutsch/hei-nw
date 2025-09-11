"""Evaluation harness and reporting utilities."""

from importlib import import_module
from typing import Any

__all__ = ["harness", "report"]


def __getattr__(name: str) -> Any:
    """Lazily import submodules to avoid eager side effects."""
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
