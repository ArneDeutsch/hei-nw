"""Manual eviction demo verifying decay and pin protection behaviour.

Run with::

    PYTHONPATH=src python -m tests.manual.eviction_demo

The script constructs a tiny store with one pinned trace and one stale
unpinned trace, then runs the eviction pass to show that only the stale
entry is removed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from hei_nw.eviction import DecayPolicy, PinProtector
from hei_nw.store import EpisodicStore


class _DemoIndex:
    """Index stub tracking basic metadata for the demo."""

    def __init__(self) -> None:
        self.meta: list[dict[str, Any]] = [
            {
                "trace_id": "pinned-trace",
                "trace": {"trace_id": "pinned-trace", "pin": True},
                "_active": True,
            },
            {
                "trace_id": "stale-trace",
                "trace": {"trace_id": "stale-trace", "pin": False},
                "_active": True,
            },
        ]

    def mark_inactive(self, trace_id: str) -> bool:
        for entry in self.meta:
            if entry.get("trace_id") == trace_id:
                entry["_active"] = False
                return True
        return False

    def update_metadata(self, trace_id: str, updates: dict[str, Any]) -> None:
        for entry in self.meta:
            if entry.get("trace_id") == trace_id:
                entry.update(updates)


def _build_demo_store() -> EpisodicStore:
    policy = DecayPolicy(
        base_ttl_seconds=30.0,
        salience_boost=0.0,
        min_ttl_seconds=30.0,
        max_ttl_seconds=3600.0,
    )
    protector = PinProtector(high_salience_floor=2.5)

    store = EpisodicStore.__new__(EpisodicStore)
    store.index = _DemoIndex()
    store.decay_policy = policy
    store.pin_protector = protector
    store._eviction_state = {}
    store._group_ids = set()
    store._hopfield_blend = 0.0
    store._hopfield_margin = 0.0
    store.max_mem_tokens = 0
    store._embed_dim = 0
    store.tokenizer = None

    base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    pinned_state = policy.create_state(
        trace_id="pinned-trace",
        score=4.0,
        pin=True,
        now=base_time,
    )

    stale_state = policy.create_state(
        trace_id="stale-trace",
        score=0.1,
        pin=False,
        now=base_time,
    )
    # Force the stale trace to expire by moving its last access into the past.
    stale_state.last_access = stale_state.last_access - (
        timedelta(seconds=stale_state.ttl_seconds) + timedelta(seconds=10)
    )

    store._eviction_state[pinned_state.trace_id] = pinned_state
    store._eviction_state[stale_state.trace_id] = stale_state

    # Update index metadata so diagnostics reflect current eviction state.
    store._update_metadata(
        pinned_state.trace_id,
        {
            "pin": pinned_state.pin,
            "salience_score": pinned_state.score,
            "last_access": pinned_state.last_access.isoformat(),
            "ttl_seconds": pinned_state.ttl_seconds,
            "expires_at": pinned_state.expires_at.isoformat(),
        },
    )
    store._update_metadata(
        stale_state.trace_id,
        {
            "pin": stale_state.pin,
            "salience_score": stale_state.score,
            "last_access": stale_state.last_access.isoformat(),
            "ttl_seconds": stale_state.ttl_seconds,
            "expires_at": stale_state.expires_at.isoformat(),
        },
    )
    return store


def main() -> None:
    store = _build_demo_store()
    print("=== Eviction Demo ===")
    print("Initial traces:")
    for trace_id, state in store._eviction_state.items():
        print(
            f"- {trace_id}: pin={state.pin}, last_access={state.last_access.isoformat()}, "
            f"expires_at={state.expires_at.isoformat()}"
        )

    eviction_time = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)
    evicted = store.evict_stale(now=eviction_time)

    print(f"\nEviction run at {eviction_time.isoformat()}")
    print(f"Evicted traces: {evicted}")
    print("\nSurviving traces:")
    for trace_id, state in store._eviction_state.items():
        print(
            f"- {trace_id}: pin={state.pin}, ttl_seconds={state.ttl_seconds}, "
            f"expires_at={state.expires_at.isoformat()}"
        )

    if "pinned-trace" in store._eviction_state and "stale-trace" not in store._eviction_state:
        print("\nResult: pinned trace survived, stale trace evicted ✅")
    else:
        print("\nResult: unexpected eviction outcome ❌")


if __name__ == "__main__":
    main()
