"""Simple text metrics: exact match, relaxed EM, token-level F1, and recall@k."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import unicodedata


def canonicalize(text: str) -> str:
    """Return a normalized version of ``text`` for relaxed comparisons."""

    stripped = text.strip().lower()
    no_punct = "".join(
        ch for ch in stripped if not unicodedata.category(ch).startswith("P")
    )
    return " ".join(no_punct.split())


def strict_em(prediction: str, truth: str) -> float:
    """Return 1.0 if *prediction* matches *truth* exactly, else 0.0."""

    return 1.0 if prediction.strip() == truth.strip() else 0.0


def relaxed_em(prediction: str, truth: str) -> float:
    """Return relaxed exact match after canonicalization."""

    return strict_em(canonicalize(prediction), canonicalize(truth))


def exact_match(prediction: str, truth: str) -> float:
    """Alias kept for backwards compatibility with strict EM."""

    return strict_em(prediction, truth)


def _tokens(text: str) -> list[str]:
    return text.strip().split()


def token_f1(prediction: str, truth: str) -> float:
    """Compute token-level F1 using simple whitespace tokenization."""
    pred_tokens = _tokens(prediction)
    truth_tokens = _tokens(truth)
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    truth_counts = Counter(truth_tokens)
    common = pred_counts & truth_counts
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(candidates: Sequence[str], truths: Sequence[str], k: int) -> float:
    """Compute recall@k for candidate strings against true answers."""
    if k <= 0:
        return 0.0
    top_k = candidates[:k]
    if not truths:
        return 0.0
    hits = len(set(top_k) & set(truths))
    return hits / len(truths)
