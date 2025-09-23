"""Compatibility helpers for typing torch modules without stubs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    class TorchModule:
        def __init__(self) -> None: ...

        def eval(self) -> TorchModule: ...

        def to(self, *args: Any, **kwargs: Any) -> TorchModule: ...

        def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

else:
    from torch.nn import Module as TorchModule

__all__ = ["TorchModule"]
