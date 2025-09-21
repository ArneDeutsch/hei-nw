from __future__ import annotations

from hei_nw.gate import NeuromodulatedGate, SalienceFeatures
from hei_nw.store import TraceWriter


def test_pointer_only_payload() -> None:
    gate = NeuromodulatedGate(threshold=0.8)
    decision = gate.decision(
        SalienceFeatures(surprise=1.2, novelty=0.6, reward=True, pin=False)
    )
    writer = TraceWriter()
    payload = writer.write(
        trace_id="trace-001",
        pointer={"doc": "chat/2025-09-10", "start": 128, "end": 160},
        entity_slots={
            "who": "Dana",
            "what": "backpack",
            "where": "Cafe Lumen",
            "when": "2025-09-10",
        },
        decision=decision,
        provenance={"source": "chat", "timestamp": "2025-09-10T01:02:03Z"},
    )

    assert payload["tokens_span_ref"] == {"doc": "chat/2025-09-10", "start": 128, "end": 160}
    assert payload["entity_slots"]["who"] == "Dana"
    assert "episode_text" not in payload
    assert payload["salience_tags"]["S"] == decision.score
    assert payload["salience_tags"]["surprise"] == decision.features.surprise
    assert payload["salience_tags"]["reward"] is True
    assert writer.records[0] == payload
    eviction = payload["eviction"]
    assert "created_at" in eviction and "expires_at" in eviction
    state = writer.eviction_state("trace-001")
    assert state is not None
    assert eviction["ttl_seconds"] == state.ttl_seconds
    assert eviction["last_access"] == eviction["created_at"]
