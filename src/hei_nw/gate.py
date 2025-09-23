"""Neuromodulated write gate computation utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "SalienceFeatures",
    "NeuromodulatedGate",
    "GateDecision",
    "surprise_from_prob",
    "surprise_from_logits",
    "novelty_from_similarity",
    "bool_to_signal",
]


_EPSILON = 1e-9


@dataclass(frozen=True)
class SalienceFeatures:
    """Container for per-episode salience signals.

    Parameters
    ----------
    surprise:
        Predictive surprise term. Expected to be ``-log p(target)`` or another
        non-negative signal where larger values indicate a mismatch between the
        model's expectation and the observed outcome.
    novelty:
        Novelty score in ``[0, 1]`` where ``1`` denotes a completely novel
        pattern and ``0`` indicates an identical match in the store.
    reward:
        Reward flag supplied by the calling environment. Set to ``True`` when
        an external reinforcement event should amplify the write decision.
    pin:
        User pin flag. Pins always receive the strongest boost in the salience
        score. By default the gate still compares the boosted score against the
        threshold, but higher-level callers (for example the evaluation harness)
        persist pinned episodes regardless of the decision. When the
        :class:`NeuromodulatedGate` is created with ``pin_override=True`` the
        threshold comparison is bypassed and ``pin=True`` yields a write
        decision directly.
    """

    surprise: float
    novelty: float
    reward: bool = False
    pin: bool = False

    def clamp(self) -> SalienceFeatures:
        """Return a new instance with values clamped to safe ranges."""

        surprise = max(0.0, float(self.surprise))
        novelty = min(max(float(self.novelty), 0.0), 1.0)
        return SalienceFeatures(
            surprise=surprise,
            novelty=novelty,
            reward=bool(self.reward),
            pin=bool(self.pin),
        )


@dataclass(frozen=True)
class GateDecision:
    """Structured response from :class:`NeuromodulatedGate`.

    Attributes
    ----------
    score:
        Weighted salience value ``S``.
    should_write:
        Whether the episode should be persisted according to the gate policy.
    contributions:
        Dictionary mapping feature names to their weighted contribution.
    features:
        Canonical, clamped :class:`SalienceFeatures` used for the decision.
    """

    score: float
    should_write: bool
    contributions: dict[str, float]
    features: SalienceFeatures


class NeuromodulatedGate:
    """Compute neuromodulated salience and threshold writes.

    The gate evaluates ``S = α·surprise + β·novelty + γ·reward + δ·pin`` and
    writes an episode when ``S > τ``. Coefficients are configurable, with design
    defaults described in :mod:`planning.design` §5.7. Set ``pin_override=True``
    to bypass the threshold whenever ``pin`` is asserted.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        delta: float = 0.8,
        threshold: float = 1.5,
        pin_override: bool = False,
    ) -> None:
        if threshold <= 0.0:
            msg = "threshold must be positive"
            raise ValueError(msg)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.pin_override = bool(pin_override)

    def score(self, features: SalienceFeatures) -> float:
        """Return salience score for *features* after clamping."""

        canonical = features.clamp()
        reward_term = self.gamma * bool_to_signal(canonical.reward)
        pin_term = self.delta * bool_to_signal(canonical.pin)
        surprise_term = self.alpha * canonical.surprise
        novelty_term = self.beta * canonical.novelty
        return surprise_term + novelty_term + reward_term + pin_term

    def decision(self, features: SalienceFeatures) -> GateDecision:
        """Return a structured gate decision for *features*."""

        canonical = features.clamp()
        surprise_term = self.alpha * canonical.surprise
        novelty_term = self.beta * canonical.novelty
        reward_term = self.gamma * bool_to_signal(canonical.reward)
        pin_term = self.delta * bool_to_signal(canonical.pin)
        total = surprise_term + novelty_term + reward_term + pin_term
        should_write = total > self.threshold or (self.pin_override and canonical.pin)
        contributions = {
            "surprise": surprise_term,
            "novelty": novelty_term,
            "reward": reward_term,
            "pin": pin_term,
        }
        return GateDecision(
            score=total,
            should_write=should_write,
            contributions=contributions,
            features=canonical,
        )

    def should_write(self, features: SalienceFeatures) -> bool:
        """Convenience wrapper returning only the write decision."""

        return self.decision(features).should_write


def surprise_from_prob(prob: float) -> float:
    """Return surprise signal ``-log(p)`` for *prob* clamped to ``(0, 1]``."""

    clamped = min(max(prob, _EPSILON), 1.0)
    return -math.log(clamped)


def surprise_from_logits(logits: Iterable[float], target_index: int) -> float:
    """Return surprise derived from raw *logits* and *target_index*.

    Parameters
    ----------
    logits:
        Iterable of unnormalised scores for each token.
    target_index:
        Index of the ground-truth token within ``logits``.
    """

    scores = list(float(logit) for logit in logits)
    if not scores:
        raise ValueError("logits must be non-empty")
    if not 0 <= target_index < len(scores):
        raise IndexError("target_index out of bounds")
    max_score = max(scores)
    shifted = [score - max_score for score in scores]
    exp_scores = [math.exp(score) for score in shifted]
    denom = sum(exp_scores)
    if denom <= 0.0:
        raise ValueError("logits sum produced zero probability")
    prob = exp_scores[target_index] / denom
    return surprise_from_prob(prob)


def novelty_from_similarity(similarity: float | None) -> float:
    """Return novelty score from ``[0, 1]`` similarity.

    ``None`` indicates absence of matching items and yields maximal novelty.
    """

    if similarity is None:
        return 1.0
    return 1.0 - min(max(float(similarity), 0.0), 1.0)


def bool_to_signal(value: bool) -> float:
    """Convert a boolean flag to ``0.0`` or ``1.0``."""

    return 1.0 if value else 0.0
