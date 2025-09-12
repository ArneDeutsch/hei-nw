"""Metric utilities for evaluation."""

from __future__ import annotations

from .compute import ComputeRecord, estimate_attention_flops, estimate_kv_bytes
from .retrieval import (
    collision_rate,
    completion_lift,
    mrr,
    near_miss_rate,
    precision_at_k,
)
from .text import exact_match, recall_at_k, token_f1
from .timing import time_block

__all__ = [
    "exact_match",
    "recall_at_k",
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
]
