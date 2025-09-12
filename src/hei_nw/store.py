from __future__ import annotations

from typing import Any

import faiss
import numpy as np


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
