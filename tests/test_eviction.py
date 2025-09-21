from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hei_nw.eviction import DecayPolicy, PinProtector
from hei_nw.store import EpisodicStore


class DummyIndex:
    def __init__(self) -> None:
        self.meta = [{"trace_id": "trace-1", "_active": True}]
        self.marked: list[str] = []

    def mark_inactive(self, trace_id: str) -> bool:
        self.marked.append(trace_id)
        for entry in self.meta:
            if entry.get("trace_id") == trace_id:
                entry["_active"] = False
        return True

    def update_metadata(self, trace_id: str, updates: dict[str, object]) -> None:
        for entry in self.meta:
            if entry.get("trace_id") == trace_id:
                entry.update(updates)

    def search(self, *_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise NotImplementedError("DummyIndex does not support search in tests")

    ef_search = 1


def _make_store(policy: DecayPolicy, protector: PinProtector, state_time: datetime) -> EpisodicStore:
    store = EpisodicStore.__new__(EpisodicStore)
    store.index = DummyIndex()
    store.decay_policy = policy
    store.pin_protector = protector
    store._eviction_state = {}
    store._group_ids = set()
    store._hopfield_blend = 0.0
    store._hopfield_margin = 0.0
    store.max_mem_tokens = 0
    store._embed_dim = 0
    store.tokenizer = None
    state = policy.create_state(
        trace_id="trace-1",
        score=0.0,
        pin=False,
        now=state_time,
    )
    store._eviction_state[state.trace_id] = state
    return store


def test_ttl_decay_removes_expired() -> None:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    policy = DecayPolicy(
        base_ttl_seconds=5.0,
        salience_boost=0.0,
        min_ttl_seconds=5.0,
        max_ttl_seconds=10.0,
    )
    protector = PinProtector()
    store = _make_store(policy, protector, base_time)
    expiry = store._eviction_state["trace-1"].expires_at + timedelta(seconds=1)

    evicted = store.evict_stale(now=expiry)

    assert evicted == ["trace-1"]
    assert "trace-1" not in store._eviction_state
    assert store.index.meta[0]["_active"] is False
    assert store.index.marked == ["trace-1"]


def test_pin_protection_blocks_eviction() -> None:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    policy = DecayPolicy(
        base_ttl_seconds=5.0,
        salience_boost=0.0,
        min_ttl_seconds=5.0,
        max_ttl_seconds=10.0,
    )
    protector = PinProtector()
    store = _make_store(policy, protector, base_time)
    store.index.meta[0]["pin"] = True
    state = store._eviction_state["trace-1"]
    store._eviction_state["trace-1"] = policy.create_state(
        trace_id=state.trace_id,
        score=state.score,
        pin=True,
        now=base_time,
    )
    expiry = store._eviction_state["trace-1"].expires_at + timedelta(seconds=1)

    evicted = store.evict_stale(now=expiry)

    assert evicted == []
    assert "trace-1" in store._eviction_state
    assert store.index.marked == []
