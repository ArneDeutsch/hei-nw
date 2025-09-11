"""Timing utilities."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType


@dataclass
class Timer:
    """Simple timer context manager recording elapsed seconds."""

    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self.start


@contextmanager
def time_block() -> Iterator[Timer]:
    """Yield a :class:`Timer` recording time spent inside the block."""
    with Timer() as t:
        yield t
