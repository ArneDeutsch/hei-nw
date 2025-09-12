from __future__ import annotations

from typing import Any

import faiss
import numpy as np
import torch
from torch import Tensor, nn


class ANNIndex:
    """Approximate nearest neighbour search over dense vectors.

    The index wraps a FAISS ``IndexHNSWFlat`` instance with an inner-product
    metric. Vectors are L2-normalized on ingestion and query, so scores
    correspond to cosine similarity.

    Parameters
    ----------
    dim:
        Dimensionality of vectors added to the index.
    m:
        HNSW graph degree. Defaults to ``32``.
    ef_search:
        Search breadth controlling recall/speed trade-off. Defaults to ``64``.
    """

    def __init__(self, dim: int, m: int = 32, ef_search: int = 64) -> None:
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.meta: list[dict[str, Any]] = []

    def add(self, vectors: np.ndarray, meta: list[dict[str, Any]]) -> None:
        """Add vectors and associated metadata to the index.

        Parameters
        ----------
        vectors:
            Array of shape ``[n, dim]``.
        meta:
            List of metadata dictionaries aligned with ``vectors``.

        Raises
        ------
        ValueError
            If ``vectors`` has wrong shape or ``meta`` length mismatches.
        """

        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            msg = f"vectors must have shape [n, {self.dim}]"
            raise ValueError(msg)
        if len(meta) != vectors.shape[0]:
            raise ValueError("meta length must match number of vectors")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.meta.extend(meta)

    def search(self, query: np.ndarray, k: int) -> list[dict[str, Any]]:
        """Return top-``k`` nearest neighbours for a query vector.

        Parameters
        ----------
        query:
            Query array of shape ``[1, dim]``.
        k:
            Number of neighbours to return.

        Returns
        -------
        list[dict]
            Metadata dictionaries augmented with a ``score`` key.
        """

        if query.dtype != np.float32:
            query = query.astype("float32")
        if query.ndim != 2 or query.shape[1] != self.dim or query.shape[0] != 1:
            msg = f"query must have shape [1, {self.dim}]"
            raise ValueError(msg)
        if k <= 0:
            raise ValueError("k must be positive")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        results: list[dict[str, Any]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < 0:
                continue
            item = dict(self.meta[idx])
            item["score"] = float(score)
            results.append(item)
        return results


class HopfieldReadout(nn.Module):
    """Inference-only modern Hopfield network readout.

    This module stores a pattern matrix ``M`` and performs a fixed number of
    query refinement steps using the modern Hopfield update rule. Parameters
    are registered as non-trainable so that forward passes do not mutate state
    or require gradients.

    Parameters
    ----------
    patterns:
        Tensor of shape ``[p, d]`` containing stored patterns.
    steps:
        Number of refinement iterations to apply. Defaults to ``1``.
    temperature:
        Softmax temperature ``T`` (``1 / beta``). Defaults to ``1.0``.
    """

    def __init__(self, patterns: Tensor, steps: int = 1, temperature: float = 1.0) -> None:
        super().__init__()
        if patterns.ndim != 2:
            raise ValueError("patterns must have shape [p, d]")
        self.patterns = nn.Parameter(patterns.clone().float(), requires_grad=False)
        self.steps = steps
        self.temperature = temperature

    def forward(
        self,
        cue: Tensor,
        candidates: Tensor | None = None,
        return_scores: bool = False,
    ) -> Tensor:
        """Refine a cue vector given stored or provided patterns.

        Parameters
        ----------
        cue:
            Query vector of shape ``[d]`` or batch ``[b, d]``.
        candidates:
            Optional pattern matrix overriding the stored ``patterns``.
        return_scores:
            If ``True``, return the attention scores instead of the refined
            query.

        Returns
        -------
        Tensor
            Refined query vector or attention scores depending on
            ``return_scores``.
        """

        patterns = candidates if candidates is not None else self.patterns
        patterns = torch.nn.functional.normalize(patterns, dim=-1)
        z = cue.float()
        squeeze = False
        if z.ndim == 1:
            z = z.unsqueeze(0)
            squeeze = True
        for _ in range(self.steps):
            z = torch.nn.functional.normalize(z, dim=-1)
            attn = torch.softmax((z @ patterns.T) / self.temperature, dim=-1)
            z = attn @ patterns
        if return_scores:
            return attn if not squeeze else attn.squeeze(0)
        return z.squeeze(0) if squeeze else z
