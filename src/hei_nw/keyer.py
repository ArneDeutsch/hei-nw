"""Sparse keyer using k-Winners-Take-All (k-WTA)."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor, nn


class KeyDict(TypedDict):
    """Sparse key representation returned by :class:`DGKeyer`."""

    indices: Tensor
    values: Tensor
    dim: int


class DGKeyer(nn.Module):
    """Project hidden states into a sparse, L1-normalized key.

    The keyer applies a linear projection followed by k-Winners-Take-All
    sparsification. Retained values keep their sign and are L1-normalized
    across the selected dimensions.

    Parameters
    ----------
    d:
        Output dimensionality of the projected key.
    k:
        Number of non-zero components to retain.
    """

    def __init__(self, d: int = 2048, k: int = 64) -> None:
        super().__init__()
        if k <= 0 or d <= 0 or k > d:
            msg = "k must be in the range (0, d] and d must be positive"
            raise ValueError(msg)
        self.d = d
        self.k = k
        # LazyLinear infers input features on first use
        self.proj = nn.LazyLinear(out_features=d, bias=False)

    def forward(self, H_t: Tensor) -> KeyDict:
        """Return sparse key for hidden states ``H_t``.

        Parameters
        ----------
        H_t:
            Hidden states with shape ``[batch, seq, hidden]``.

        Returns
        -------
        KeyDict
            Dictionary containing ``indices`` and ``values`` tensors and the
            dense dimensionality ``dim``.
        """

        pooled = H_t.mean(dim=1)
        q = self.proj(pooled)
        magnitudes = q.abs()
        topk = magnitudes.topk(self.k, dim=-1)
        indices = topk.indices
        values = q.gather(-1, indices)
        l1 = values.abs().sum(dim=-1, keepdim=True)
        values = values / l1.clamp_min(torch.finfo(values.dtype).eps)
        return {"indices": indices, "values": values, "dim": self.d}


def to_dense(key: KeyDict, dim: int | None = None) -> Tensor:
    """Convert a sparse key representation into a dense vector.

    Parameters
    ----------
    key:
        Sparse key with ``indices`` and ``values``.
    dim:
        Target dimensionality. Defaults to ``key['dim']``.

    Returns
    -------
    torch.Tensor
        Dense float32 tensor of shape ``[batch, dim]``.
    """

    indices = key["indices"]
    values = key["values"].to(torch.float32)
    if dim is None:
        dim = int(key["dim"])
    dense = torch.zeros(indices.shape[0], dim, dtype=torch.float32, device=values.device)
    dense.scatter_(1, indices, values)
    return dense
