"""Vector store components for episodic retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import faiss
import numpy as np
import torch
from torch import Tensor, nn

from .keyer import DGKeyer, to_dense

__all__ = ["ANNIndex", "HopfieldReadout", "EpisodicStore"]


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

    def __init__(self, dim: int, m: int = 32, ef_search: int = 64) -> None:  # noqa: ARG002
        """Create an empty index of dimensionality ``dim``."""

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
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
        """Initialise the readout with stored ``patterns``."""

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


class EpisodicStore:
    """Associative store combining a keyer, ANN index, and Hopfield readout.

    The store builds dense keys for episodes using :class:`DGKeyer`, indexes
    them with :class:`ANNIndex`, and optionally refines queries via
    :class:`HopfieldReadout`. It also tracks simple near-miss and collision
    diagnostics when ground-truth labels are supplied.

    Parameters
    ----------
    keyer:
        Keyer used to produce sparse keys.
    index:
        Approximate nearest neighbour index over dense keys.
    hopfield:
        Modern Hopfield readout used for query refinement.
    tokenizer:
        Tokenizer used to split text for hashing.
    vectors:
        Dense key vectors stored in the index.
    group_ids:
        Set of group ids present in the store.
    embed_dim:
        Dimensionality of the hashed embedding space.
    max_mem_tokens:
        Maximum number of tokens allowed when packing traces (unused in M2-T4
        but stored for future use).
    """

    def __init__(
        self,
        keyer: DGKeyer,
        index: ANNIndex,
        hopfield: HopfieldReadout,
        tokenizer: Any,
        vectors: list[np.ndarray],
        group_ids: set[int],
        embed_dim: int,
        max_mem_tokens: int,
    ) -> None:
        """Initialise the store with precomputed components."""

        self.keyer = keyer
        self.index = index
        self.hopfield = hopfield
        self.tokenizer = tokenizer
        self._vectors = vectors
        self._group_ids = group_ids
        self._embed_dim = embed_dim
        self.max_mem_tokens = max_mem_tokens

    @staticmethod
    def _hash_embed(text: str, tokenizer: Any, dim: int) -> Tensor:
        """Return deterministic hashed embedding for *text*.

        The embedding sums one-hot vectors derived from token hashes to
        produce a simple bag-of-words representation.
        """

        tokens = tokenizer.tokenize(text) if hasattr(tokenizer, "tokenize") else text.split()
        vec = torch.zeros(1, 1, dim, dtype=torch.float32)
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % dim
            vec[0, 0, h] += 1.0
        return vec

    @classmethod
    def from_records(
        cls,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_mem_tokens: int,
        *,
        embed_dim: int = 64,
        hopfield_steps: int = 1,
        hopfield_temperature: float = 1.0,
        keyer: DGKeyer | None = None,
    ) -> EpisodicStore:
        """Build a store from Scenario A-style records.

        Only records with ``should_remember=True`` are indexed. For each such
        record we compute a dense key, attach metadata, and add it to the ANN
        index. The resulting dense keys are also used to initialise the
        Hopfield readout. Callers may override the number of refinement steps
        and the softmax temperature applied by the Hopfield module via
        ``hopfield_steps`` and ``hopfield_temperature`` respectively. A
        custom :class:`DGKeyer` can be supplied via ``keyer`` to control
        sparsity of the dense keys.
        """

        keyer_module = keyer if keyer is not None else DGKeyer()
        vectors: list[np.ndarray] = []
        meta: list[dict[str, Any]] = []
        for rec in records:
            if not bool(rec.get("should_remember")):
                continue
            H = cls._hash_embed(str(rec["episode_text"]), tokenizer, embed_dim)
            key = keyer_module(H)
            dense = to_dense(key).squeeze(0).detach().cpu().numpy()
            trace = {
                "group_id": rec["group_id"],
                "answers": rec["answers"],
                "episode_text": rec["episode_text"],
            }
            meta.append(
                {
                    "group_id": rec["group_id"],
                    "answers": rec["answers"],
                    "trace": trace,
                    "should_remember": rec["should_remember"],
                    "key_vector": dense,
                }
            )
            vectors.append(dense)
        index = ANNIndex(dim=keyer_module.d)
        if vectors:
            vec_array = np.stack(vectors).astype("float32")
            index.add(vec_array, meta)
            patterns = torch.from_numpy(vec_array)
        else:
            patterns = torch.zeros(1, keyer_module.d, dtype=torch.float32)
        hopfield = HopfieldReadout(
            patterns, steps=hopfield_steps, temperature=hopfield_temperature
        )
        group_ids = {m["group_id"] for m in meta}
        return cls(
            keyer_module,
            index,
            hopfield,
            tokenizer,
            vectors,
            group_ids,
            embed_dim,
            max_mem_tokens,
        )

    def _embed(self, text: str) -> Tensor:
        return self._hash_embed(text, self.tokenizer, self._embed_dim)

    def query(
        self,
        cue_text: str,
        top_k_candidates: int = 64,
        return_m: int = 4,
        use_hopfield: bool = True,
        *,
        group_id: int | None = None,
        should_remember: bool | None = None,
    ) -> dict[str, Any]:
        """Query the store with *cue_text*.

        Parameters
        ----------
        cue_text:
            Natural-language cue describing an episode.
        top_k_candidates:
            Number of ANN neighbours to consider.
        return_m:
            Number of traces to return.
        use_hopfield:
            Whether to refine the query with Hopfield attention over
            candidates.
        group_id, should_remember:
            Optional ground-truth labels enabling near-miss and collision
            diagnostics.
        """

        H = self._embed(cue_text)
        key = self.keyer(H)
        dense = to_dense(key).detach().cpu().numpy()
        results = self.index.search(dense, k=top_k_candidates)
        if not results:
            diagnostics = {
                "near_miss": False,
                "collision": bool(group_id in self._group_ids) if group_id is not None else False,
            }
            return {"selected": [], "candidates": [], "diagnostics": diagnostics}
        if use_hopfield:
            cand_vecs = torch.from_numpy(
                np.stack([r["key_vector"] for r in results]).astype("float32")
            )
            scores = self.hopfield(
                torch.from_numpy(dense.squeeze(0)), candidates=cand_vecs, return_scores=True
            )
            top_idx = torch.topk(scores, min(return_m, len(results))).indices.tolist()
        else:
            top_idx = list(range(min(return_m, len(results))))
        selected = [results[i]["trace"] for i in top_idx]
        candidates: list[dict[str, Any]] = []
        for r in results:
            r_copy = {k: v for k, v in r.items() if k != "key_vector"}
            candidates.append(r_copy)
        top_group = results[0]["group_id"]
        near_miss = group_id is not None and should_remember is False and top_group == group_id
        collision = (
            group_id is not None
            and should_remember is True
            and top_group != group_id
            and group_id in self._group_ids
        )
        diagnostics = {"near_miss": bool(near_miss), "collision": bool(collision)}
        return {"selected": selected, "candidates": candidates, "diagnostics": diagnostics}
