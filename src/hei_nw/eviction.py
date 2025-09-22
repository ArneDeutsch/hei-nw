"""Decay and eviction utilities for the episodic trace store."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

__all__ = ["TraceEvictionState", "DecayPolicy", "PinProtector"]


@dataclass
class TraceEvictionState:
    """Stateful metadata tracked for eviction/decay decisions."""

    trace_id: str
    score: float
    pin: bool
    created_at: datetime
    last_access: datetime
    ttl_seconds: float

    @property
    def expires_at(self) -> datetime:
        """Return the timestamp when the trace should expire."""

        return self.last_access + timedelta(seconds=self.ttl_seconds)


class DecayPolicy:
    """Schedule trace expiration based on age, use, and salience."""

    def __init__(
        self,
        *,
        base_ttl_seconds: float = 30 * 24 * 60 * 60,
        salience_boost: float = 0.35,
        min_ttl_seconds: float = 6 * 60 * 60,
        max_ttl_seconds: float = 90 * 24 * 60 * 60,
    ) -> None:
        if base_ttl_seconds <= 0:
            msg = "base_ttl_seconds must be positive"
            raise ValueError(msg)
        if min_ttl_seconds <= 0:
            msg = "min_ttl_seconds must be positive"
            raise ValueError(msg)
        if max_ttl_seconds < min_ttl_seconds:
            msg = "max_ttl_seconds must be >= min_ttl_seconds"
            raise ValueError(msg)
        self.base_ttl_seconds = float(base_ttl_seconds)
        self.salience_boost = float(salience_boost)
        self.min_ttl_seconds = float(min_ttl_seconds)
        self.max_ttl_seconds = float(max_ttl_seconds)

    def create_state(
        self,
        *,
        trace_id: str,
        score: float,
        pin: bool,
        now: datetime | None = None,
    ) -> TraceEvictionState:
        """Return initial eviction state for a new trace."""

        instant = self._ensure_time(now)
        ttl = self._compute_ttl(score=score, pin=pin)
        return TraceEvictionState(
            trace_id=str(trace_id),
            score=float(score),
            pin=bool(pin),
            created_at=instant,
            last_access=instant,
            ttl_seconds=ttl,
        )

    def on_access(
        self,
        state: TraceEvictionState,
        *,
        score: float | None = None,
        now: datetime | None = None,
    ) -> TraceEvictionState:
        """Update *state* when a trace is accessed."""

        instant = self._ensure_time(now)
        state.last_access = instant
        if score is not None:
            state.score = float(score)
            state.ttl_seconds = self._compute_ttl(score=state.score, pin=state.pin)
        return state

    def should_evict(
        self,
        state: TraceEvictionState,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Return ``True`` if *state* has expired at *now*."""

        instant = self._ensure_time(now)
        return instant >= state.expires_at

    def _compute_ttl(self, *, score: float, pin: bool) -> float:
        if pin:
            return self.max_ttl_seconds
        boost = max(float(score), 0.0)
        ttl = self.base_ttl_seconds * (1.0 + self.salience_boost * boost)
        ttl = max(ttl, self.min_ttl_seconds)
        ttl = min(ttl, self.max_ttl_seconds)
        return float(ttl)

    @staticmethod
    def _ensure_time(value: datetime | None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class PinProtector:
    """Guard against evicting pinned or very salient traces."""

    def __init__(self, *, high_salience_floor: float = 3.0, enabled: bool = True) -> None:
        self.high_salience_floor = float(high_salience_floor)
        self.enabled = bool(enabled)

    def blocks_eviction(self, state: TraceEvictionState) -> bool:
        """Return ``True`` if eviction should be blocked for *state*."""

        if not self.enabled:
            return False
        if state.pin:
            return True
        return state.score >= self.high_salience_floor

    def batch_filter(self, states: Iterable[TraceEvictionState]) -> list[TraceEvictionState]:
        """Return list of states allowed to proceed to eviction checks."""

        return [state for state in states if not self.blocks_eviction(state)]
