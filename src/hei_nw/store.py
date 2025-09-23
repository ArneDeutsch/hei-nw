"""Vector store components for episodic retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from typing import Any, TypeAlias, cast

import faiss
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .eviction import DecayPolicy, PinProtector, TraceEvictionState
from .gate import GateDecision
from .keyer import DGKeyer, to_dense
from .utils.torch_types import TorchModule

__all__ = ["ANNIndex", "HopfieldReadout", "EpisodicStore", "TraceWriter"]


_POINTER_KEYS = {"doc", "start", "end"}
_BANNED_TEXT_KEYS = {"episode_text", "raw_text", "snippet", "full_text", "text"}


FloatArray: TypeAlias = NDArray[np.float32]


class TraceWriter:
    """Persist pointer-only episodic traces with salience and decay metadata."""

    def __init__(self, *, decay_policy: DecayPolicy | None = None) -> None:
        self._records: list[dict[str, Any]] = []
        self._decay_policy = decay_policy or DecayPolicy()
        self._eviction_state: dict[str, TraceEvictionState] = {}

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return a shallow copy of persisted trace payloads."""

        return list(self._records)

    def eviction_state(self, trace_id: str) -> TraceEvictionState | None:
        """Return eviction metadata for *trace_id* if available."""

        return self._eviction_state.get(str(trace_id))

    def write(
        self,
        *,
        trace_id: str,
        pointer: Mapping[str, Any],
        entity_slots: Mapping[str, Any],
        decision: GateDecision,
        provenance: Mapping[str, Any] | None = None,
        extras: Mapping[str, Any] | None = None,
        written_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Persist a pointer-only trace and return the stored payload."""

        timestamp = self._now(written_at)
        state = self._decay_policy.create_state(
            trace_id=str(trace_id),
            score=decision.score,
            pin=decision.features.pin,
            now=timestamp,
        )
        self._eviction_state[state.trace_id] = state

        payload: dict[str, Any] = {
            "trace_id": str(trace_id),
            "tokens_span_ref": self._normalise_pointer(pointer),
            "entity_slots": self._normalise_slots(entity_slots),
            "salience_tags": self._salience_tags(decision),
            "eviction": {
                "ttl_seconds": state.ttl_seconds,
                "created_at": self._isoformat(state.created_at),
                "last_access": self._isoformat(state.last_access),
                "expires_at": self._isoformat(state.expires_at),
            },
        }
        if provenance is not None:
            payload["provenance"] = self._normalise_provenance(provenance)
        if extras is not None:
            payload["extras"] = self._normalise_extras(extras)
        self._records.append(payload)
        return payload

    @staticmethod
    def _now(value: datetime | None = None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _isoformat(cls, value: datetime) -> str:
        return cls._now(value).isoformat()

    @staticmethod
    def _normalise_pointer(pointer: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(pointer, Mapping):
            raise TypeError("pointer must be a mapping")
        unknown = set(pointer.keys()) - _POINTER_KEYS
        if unknown:
            msg = f"pointer contains unsupported fields: {sorted(unknown)}"
            raise ValueError(msg)
        doc = str(pointer.get("doc", "")).strip()
        if not doc:
            raise ValueError("pointer.doc must be a non-empty string")
        start_raw = pointer.get("start")
        end_raw = pointer.get("end")
        if start_raw is None or end_raw is None:
            raise ValueError("pointer.start and pointer.end must be provided")
        try:
            start = int(start_raw)
            end = int(end_raw)
        except (TypeError, ValueError) as exc:  # pragma: no cover - sanity guard
            raise ValueError("pointer.start and pointer.end must be integers") from exc
        if start < 0 or end < 0:
            raise ValueError("pointer offsets must be non-negative")
        if end <= start:
            raise ValueError("pointer.end must be greater than pointer.start")
        return {"doc": doc, "start": start, "end": end}

    @staticmethod
    def _normalise_slots(slots: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(slots, Mapping):
            raise TypeError("entity_slots must be a mapping")
        for banned in _BANNED_TEXT_KEYS:
            if banned in slots:
                raise ValueError(f"entity_slots cannot contain raw text key '{banned}'")
        normalised: dict[str, Any] = {
            key: str(slots.get(key, "") or "").strip() for key in ("who", "what", "where", "when")
        }
        extras = slots.get("extras")
        if extras is not None:
            if not isinstance(extras, Mapping):
                raise TypeError("entity_slots['extras'] must be a mapping if provided")
            for banned in _BANNED_TEXT_KEYS:
                if banned in extras:
                    raise ValueError(
                        f"entity_slots['extras'] cannot contain raw text key '{banned}'"
                    )
            normalised["extras"] = {str(k): str(v) for k, v in extras.items() if v is not None}
        return normalised

    @staticmethod
    def _salience_tags(decision: GateDecision) -> dict[str, Any]:
        features = decision.features
        tags: dict[str, Any] = {
            "surprise": float(features.surprise),
            "novelty": float(features.novelty),
            "reward": bool(features.reward),
            "pin": bool(features.pin),
            "S": float(decision.score),
        }
        if decision.contributions:
            tags["contributions"] = {k: float(v) for k, v in decision.contributions.items()}
        return tags

    @staticmethod
    def _normalise_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        allowed = {"source", "timestamp", "confidence"}
        unknown = set(provenance.keys()) - allowed
        if unknown:
            msg = f"provenance contains unsupported fields: {sorted(unknown)}"
            raise ValueError(msg)
        source = str(provenance.get("source", "")).strip()
        if not source:
            raise ValueError("provenance.source must be provided")
        timestamp = str(provenance.get("timestamp", "")).strip()
        if not timestamp:
            raise ValueError("provenance.timestamp must be provided")
        confidence_val = provenance.get("confidence")
        confidence: float | None
        if confidence_val is None:
            confidence = None
        else:
            try:
                confidence = float(confidence_val)
            except (TypeError, ValueError) as exc:  # pragma: no cover - sanity guard
                raise ValueError("provenance.confidence must be numeric") from exc
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("provenance.confidence must lie in [0, 1]")
        result: dict[str, Any] = {"source": source, "timestamp": timestamp}
        if confidence is not None:
            result["confidence"] = confidence
        return result

    @staticmethod
    def _normalise_extras(extras: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(extras, Mapping):
            raise TypeError("extras must be a mapping")
        for banned in _BANNED_TEXT_KEYS:
            if banned in extras:
                raise ValueError(f"extras cannot contain raw text key '{banned}'")
        return {str(k): str(v) for k, v in extras.items()}


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
    ef_construction:
        Construction breadth controlling graph quality. Defaults to ``200``.
    ef_search:
        Search breadth controlling recall/speed trade-off. Defaults to ``64``.
    """

    def __init__(
        self,
        dim: int,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        """Create an empty index of dimensionality ``dim``."""

        if dim <= 0:
            msg = "dim must be positive"
            raise ValueError(msg)
        if m <= 0:
            msg = "m must be positive"
            raise ValueError(msg)
        if ef_construction <= 0:
            msg = "ef_construction must be positive"
            raise ValueError(msg)
        if ef_search <= 0:
            msg = "ef_search must be positive"
            raise ValueError(msg)

        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.index = faiss.IndexHNSWFlat(dim, m)
        # Configure construction/search breadth per design defaults.
        self.index.hnsw.efConstruction = ef_construction
        self.set_ef_search(ef_search)
        self.meta: list[dict[str, Any]] = []
        self._trace_slots: dict[str, set[int]] = {}

    @property
    def ef_search(self) -> int:
        """Return the current search breadth."""

        return int(self.index.hnsw.efSearch)

    def set_ef_search(self, ef_search: int) -> None:
        """Update the search breadth controlling recall/speed trade-offs."""

        if ef_search <= 0:
            msg = "ef_search must be positive"
            raise ValueError(msg)
        self.index.hnsw.efSearch = int(ef_search)

    def add(self, vectors: FloatArray, meta: list[dict[str, Any]]) -> None:
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

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            msg = f"vectors must have shape [n, {self.dim}]"
            raise ValueError(msg)
        if len(meta) != vectors.shape[0]:
            raise ValueError("meta length must match number of vectors")
        if vectors.shape[0] == 0:
            return
        vectors = cast(FloatArray, np.ascontiguousarray(vectors, dtype="float32"))
        faiss.normalize_L2(vectors)
        start = len(self.meta)
        enriched: list[dict[str, Any]] = []
        for offset, item in enumerate(meta):
            entry = dict(item)
            trace_id = entry.get("trace_id")
            if trace_id is None:
                trace = entry.get("trace")
                if isinstance(trace, Mapping) and trace.get("trace_id") is not None:
                    trace_id = trace["trace_id"]
                elif entry.get("group_id") is not None:
                    trace_id = f"group-{entry['group_id']}"
                else:
                    trace_id = f"trace-{start + offset}"
            trace_id = str(trace_id)
            slot = start + offset
            entry["trace_id"] = trace_id
            entry["_slot"] = slot
            entry["_active"] = True
            enriched.append(entry)
            slots = self._trace_slots.setdefault(trace_id, set())
            slots.add(slot)
        self.index.add(vectors)
        self.meta.extend(enriched)

    def search(self, query: FloatArray, k: int) -> list[dict[str, Any]]:
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

        if query.ndim != 2 or query.shape[1] != self.dim or query.shape[0] != 1:
            msg = f"query must have shape [1, {self.dim}]"
            raise ValueError(msg)
        if k <= 0:
            raise ValueError("k must be positive")
        total = self.index.ntotal
        if total == 0:
            return []
        if k > self.ef_search:
            msg = f"k ({k}) cannot exceed ef_search ({self.ef_search})"
            raise ValueError(msg)
        effective_k = min(k, total)
        query = cast(FloatArray, np.ascontiguousarray(query, dtype="float32"))
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, effective_k)
        ranked: list[dict[str, Any]] = []
        for idx, distance in zip(indices[0], scores[0], strict=False):
            if idx < 0 or idx >= len(self.meta):
                continue
            meta_entry = self.meta[idx]
            if not meta_entry.get("_active", True):
                continue
            item = dict(meta_entry)
            item.pop("_slot", None)
            item.pop("_active", None)
            sim = -float(distance)
            item["score"] = sim
            item["distance"] = float(distance)
            ranked.append(item)
        ranked.sort(key=lambda entry: entry["score"], reverse=True)
        return ranked

    def mark_inactive(self, trace_id: str) -> bool:
        """Mark all entries for *trace_id* as inactive."""

        key = str(trace_id)
        slots = list(self._trace_slots.get(key, ()))
        if not slots:
            return False
        removed = False
        for slot in slots:
            if 0 <= slot < len(self.meta):
                entry = self.meta[slot]
                if entry.get("_active", True):
                    entry["_active"] = False
                    removed = True
        if removed:
            self._trace_slots.pop(key, None)
        return removed

    def update_metadata(self, trace_id: str, updates: Mapping[str, Any]) -> None:
        """Update stored metadata for *trace_id* with *updates*."""

        key = str(trace_id)
        for entry in self.meta:
            if entry.get("trace_id") == key:
                entry.update(updates)


class HopfieldReadout(TorchModule):
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
        vectors: list[FloatArray],
        group_ids: set[int],
        embed_dim: int,
        max_mem_tokens: int,
        decay_policy: DecayPolicy | None = None,
        pin_protector: PinProtector | None = None,
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
        self._hopfield_blend = 0.2
        self._hopfield_margin = 0.01
        self.decay_policy = decay_policy or DecayPolicy()
        self.pin_protector = pin_protector or PinProtector()
        self._eviction_state: dict[str, TraceEvictionState] = {}
        self._bootstrap_eviction_state()

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
        ann_m: int = 32,
        ann_ef_construction: int = 200,
        ann_ef_search: int = 64,
        decay_policy: DecayPolicy | None = None,
        pin_protector: PinProtector | None = None,
    ) -> EpisodicStore:
        """Build a store from Scenario A-style records.

        Only records with ``should_remember=True`` are indexed. For each such
        record we compute a dense key, attach metadata, and add it to the ANN
        index. The resulting dense keys are also used to initialise the
        Hopfield readout. Callers may override the number of refinement steps
        and the softmax temperature applied by the Hopfield module via
        ``hopfield_steps`` and ``hopfield_temperature`` respectively. A
        custom :class:`DGKeyer` can be supplied via ``keyer`` to control
        sparsity of the dense keys. The HNSW configuration can be tuned via
        ``ann_m``, ``ann_ef_construction``, and ``ann_ef_search``.
        """

        keyer_module = keyer if keyer is not None else DGKeyer()
        vectors: list[FloatArray] = []
        meta: list[dict[str, Any]] = []
        for rec in records:
            if not bool(rec.get("should_remember")):
                continue
            H = cls._hash_embed(str(rec["episode_text"]), tokenizer, embed_dim)
            key = keyer_module(H)
            dense = to_dense(key).squeeze(0).detach().cpu().numpy().astype("float32", copy=False)
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
        index = ANNIndex(
            dim=keyer_module.d,
            m=ann_m,
            ef_construction=ann_ef_construction,
            ef_search=ann_ef_search,
        )
        if vectors:
            vec_array = np.stack(vectors).astype("float32")
            index.add(vec_array, meta)
            patterns = torch.from_numpy(vec_array)
        else:
            patterns = torch.zeros(1, keyer_module.d, dtype=torch.float32)
        hopfield = HopfieldReadout(patterns, steps=hopfield_steps, temperature=hopfield_temperature)
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
            decay_policy=decay_policy,
            pin_protector=pin_protector,
        )

    def _embed(self, text: str) -> Tensor:
        return self._hash_embed(text, self.tokenizer, self._embed_dim)

    def _bootstrap_eviction_state(self) -> None:
        now = self._normalize_time()
        meta = getattr(self.index, "meta", None)
        if not isinstance(meta, list):
            return
        for entry in meta:
            if not isinstance(entry, Mapping):
                continue
            if not entry.get("_active", True):
                continue
            trace_id = str(entry.get("trace_id"))
            if trace_id in self._eviction_state:
                continue
            pin_flag = bool(entry.get("pin", False))
            trace_info = entry.get("trace")
            if isinstance(trace_info, Mapping):
                pin_flag = bool(trace_info.get("pin", pin_flag))
            score_val = float(entry.get("salience_score", 0.0))
            state = self.decay_policy.create_state(
                trace_id=trace_id,
                score=score_val,
                pin=pin_flag,
                now=now,
            )
            self._eviction_state[trace_id] = state
            self._update_metadata(
                trace_id,
                {
                    "pin": pin_flag,
                    "salience_score": state.score,
                    "last_access": state.last_access.isoformat(),
                    "ttl_seconds": state.ttl_seconds,
                    "expires_at": state.expires_at.isoformat(),
                },
            )

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
        dense = to_dense(key).detach().cpu().numpy().astype("float32", copy=False)
        ef_search = getattr(self.index, "ef_search", top_k_candidates)
        k = min(top_k_candidates, int(ef_search))
        results = self.index.search(dense, k=k)
        if not results:
            diagnostics = {
                "near_miss": False,
                "collision": bool(group_id in self._group_ids) if group_id is not None else False,
                "pre_top1_group": None,
                "post_top1_group": None,
                "rank_delta": 0,
            }
            return {
                "selected": [],
                "candidates": [],
                "diagnostics": diagnostics,
                "baseline_candidates": [],
                "baseline_diagnostics": diagnostics,
            }
        baseline_indices = list(range(len(results)))
        baseline_top1_group = results[0]["group_id"]
        pre_top1_group = baseline_top1_group
        pre_rank: int | None = None
        if group_id is not None:
            for idx, res in enumerate(results):
                if res.get("group_id") == group_id:
                    pre_rank = idx
                    break
        order: torch.Tensor
        if use_hopfield:
            cand_vecs = torch.from_numpy(
                np.stack([r["key_vector"] for r in results]).astype("float32")
            )
            baseline_scores = torch.tensor(
                [float(r.get("score", 0.0)) for r in results], dtype=torch.float32
            )
            cue_tensor = torch.from_numpy(dense.squeeze(0)).to(torch.float32)
            refined = self.hopfield(cue_tensor)
            refined = torch.nn.functional.normalize(refined, dim=-1)
            normalized_candidates = torch.nn.functional.normalize(cand_vecs, dim=-1)
            cos_scores = normalized_candidates @ refined.unsqueeze(-1)
            cos_scores = cos_scores.squeeze(-1)

            def _softmax_norm(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.numel() == 0:
                    return tensor
                shifted = tensor - tensor.max()
                return torch.softmax(shifted, dim=-1)

            baseline_norm = _softmax_norm(baseline_scores)
            hopfield_norm = _softmax_norm(cos_scores / max(self.hopfield.temperature, 1e-6))
            hopfield_scores = (
                1.0 - self._hopfield_blend
            ) * baseline_norm + self._hopfield_blend * hopfield_norm
            order = torch.argsort(hopfield_scores, descending=True)

            if order.numel():
                top_index = int(order[0])
                baseline_idx = 0
                baseline_ref = baseline_norm[baseline_idx] if baseline_norm.numel() else 0.0
                hopfield_ref = hopfield_scores[top_index]
                if (
                    top_index != baseline_idx
                    and hopfield_ref <= baseline_ref + self._hopfield_margin
                ):
                    order = torch.arange(len(results))
        else:
            order = torch.arange(len(results))
        if use_hopfield and order.numel():
            baseline_order = list(range(len(results)))
            top_index = int(order[0])
            if top_index != baseline_order[0]:
                order_list = [top_index] + [idx for idx in baseline_order if idx != top_index]
            else:
                order_list = baseline_order
        else:
            order_list = [int(idx) for idx in order.tolist()]
        if not order_list:
            order_list = list(range(len(results)))
        ordered_results = [results[i] for i in order_list]
        top_k = min(return_m, len(order_list))
        top_indices = order_list[:top_k]
        post_top1_group = ordered_results[0]["group_id"] if ordered_results else baseline_top1_group
        post_rank: int | None = None
        if pre_rank is not None:
            try:
                post_rank = order_list.index(pre_rank)
            except ValueError:
                post_rank = None
        rank_delta = 0
        if pre_rank is not None and post_rank is not None:
            rank_delta = pre_rank - post_rank
        selected = [results[i]["trace"] for i in top_indices]
        accessed_states: dict[str, TraceEvictionState] = {}
        for idx in top_indices:
            res = results[idx]
            trace_id = res.get("trace_id")
            if trace_id is None:
                continue
            state = self._touch_trace(str(trace_id), float(res.get("score", 0.0)))
            accessed_states[str(trace_id)] = state
        candidates: list[dict[str, Any]] = []
        for idx in order_list:
            r = results[idx]
            r_copy = {k: v for k, v in r.items() if k != "key_vector"}
            cached_state = accessed_states.get(str(r.get("trace_id")))
            if cached_state is not None:
                r_copy["last_access"] = cached_state.last_access.isoformat()
                r_copy["ttl_seconds"] = cached_state.ttl_seconds
                r_copy["expires_at"] = cached_state.expires_at.isoformat()
            candidates.append(r_copy)
        baseline_candidates: list[dict[str, Any]] = []
        for idx in baseline_indices:
            r = results[idx]
            r_copy = {k: v for k, v in r.items() if k != "key_vector"}
            baseline_candidates.append(r_copy)
        top_group_final = candidates[0]["group_id"] if candidates else baseline_top1_group
        baseline_near_miss = (
            group_id is not None and should_remember is False and baseline_top1_group == group_id
        )
        baseline_collision = (
            group_id is not None
            and should_remember is True
            and baseline_top1_group != group_id
            and group_id in self._group_ids
        )
        near_miss = (
            group_id is not None and should_remember is False and top_group_final == group_id
        )
        collision = (
            group_id is not None
            and should_remember is True
            and top_group_final != group_id
            and group_id in self._group_ids
        )
        diagnostics = {
            "near_miss": bool(near_miss),
            "collision": bool(collision),
            "pre_top1_group": pre_top1_group,
            "post_top1_group": post_top1_group,
            "rank_delta": rank_delta,
        }
        baseline_diagnostics = {
            "near_miss": bool(baseline_near_miss),
            "collision": bool(baseline_collision),
            "pre_top1_group": baseline_top1_group,
            "post_top1_group": baseline_top1_group,
            "rank_delta": 0,
        }
        return {
            "selected": selected,
            "candidates": candidates,
            "diagnostics": diagnostics,
            "baseline_candidates": baseline_candidates,
            "baseline_diagnostics": baseline_diagnostics,
        }

    def evict_stale(self, *, now: datetime | None = None) -> list[str]:
        """Evict expired traces and return their ids."""

        instant = self._normalize_time(now)
        removed: list[str] = []
        for trace_id, state in list(self._eviction_state.items()):
            if self.pin_protector.blocks_eviction(state):
                continue
            if self.decay_policy.should_evict(state, now=instant):
                removed.append(trace_id)
                if hasattr(self.index, "mark_inactive"):
                    self.index.mark_inactive(trace_id)
                else:
                    meta = getattr(self.index, "meta", None)
                    if isinstance(meta, list):
                        for entry in meta:
                            if (
                                isinstance(entry, MutableMapping)
                                and entry.get("trace_id") == trace_id
                            ):
                                entry["_active"] = False
                self._eviction_state.pop(trace_id, None)
        return removed

    def _touch_trace(self, trace_id: str, score: float) -> TraceEvictionState:
        state = self._eviction_state.get(trace_id)
        instant = self._normalize_time()
        if state is None:
            pin_flag = self._pin_for_trace(trace_id)
            state = self.decay_policy.create_state(
                trace_id=trace_id,
                score=score,
                pin=pin_flag,
                now=instant,
            )
        state = self.decay_policy.on_access(state, score=score, now=instant)
        self._eviction_state[trace_id] = state
        self._update_metadata(
            trace_id,
            {
                "pin": state.pin,
                "salience_score": state.score,
                "last_access": state.last_access.isoformat(),
                "ttl_seconds": state.ttl_seconds,
                "expires_at": state.expires_at.isoformat(),
            },
        )
        return state

    def _pin_for_trace(self, trace_id: str) -> bool:
        meta = getattr(self.index, "meta", None)
        if not isinstance(meta, list):
            return False
        for entry in meta:
            if not isinstance(entry, MutableMapping):
                continue
            if entry.get("trace_id") == trace_id:
                trace_info = entry.get("trace")
                if isinstance(trace_info, Mapping) and trace_info.get("pin") is not None:
                    return bool(trace_info.get("pin"))
                return bool(entry.get("pin", False))
        return False

    def _update_metadata(self, trace_id: str, updates: Mapping[str, Any]) -> None:
        payload = dict(updates)
        if hasattr(self.index, "update_metadata"):
            self.index.update_metadata(trace_id, payload)
        meta = getattr(self.index, "meta", None)
        if not isinstance(meta, list):
            return
        for entry in meta:
            if isinstance(entry, MutableMapping) and entry.get("trace_id") == trace_id:
                entry.update(payload)

    @staticmethod
    def _normalize_time(value: datetime | None = None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
