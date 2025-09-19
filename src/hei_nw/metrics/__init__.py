"""Metric utilities for evaluation."""

from __future__ import annotations

from .compute import ComputeRecord, estimate_attention_flops, estimate_kv_bytes
from .retrieval import (
    collision_rate,
    completion_lift,
    hopfield_rank_improved_rate,
    mrr,
    near_miss_rate,
    precision_at_k,
)
from .text import canonicalize, exact_match, recall_at_k, relaxed_em, strict_em, token_f1
from .timing import time_block

__all__ = [
    "canonicalize",
    "exact_match",
    "recall_at_k",
    "relaxed_em",
    "strict_em",
    "token_f1",
    "time_block",
    "estimate_attention_flops",
    "estimate_kv_bytes",
    "ComputeRecord",
    "precision_at_k",
    "mrr",
    "near_miss_rate",
    "collision_rate",
    "completion_lift",
    "hopfield_rank_improved_rate",
]
