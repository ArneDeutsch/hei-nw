"""Utilities for setting global random seeds."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int) -> None:
    """Set the seed for `random`, `numpy`, and `torch` if available."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
