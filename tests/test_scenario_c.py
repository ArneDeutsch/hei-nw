from __future__ import annotations

import json
from pathlib import Path

import pytest

from hei_nw.datasets.scenario_c import generate

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scenario_c_gate.json"


def test_reward_pin_annotations() -> None:
    payload = json.loads(FIXTURE_PATH.read_text())
    seed = payload["seed"]
    n = payload["n"]
    records = generate(n, seed=seed)

    for expectation in payload["expectations"]:
        record = records[expectation["index"]]
        gate_features = record["gate_features"]
        assert record["server"] == expectation["server"]
        assert record["event_type"] == expectation["event_type"]
        assert record["should_remember"] is expectation["should_remember"]
        assert gate_features["reward"] is expectation["reward"]
        assert gate_features["pin"] is expectation["pin"]
        assert record["reward_annotation"] == expectation["reward_annotation"]
        assert record["pin_annotation"] == expectation["pin_annotation"]
        assert record["novelty_counters"] == expectation["novelty_counters"]
        assert gate_features["novelty"] == pytest.approx(expectation["novelty"], rel=1e-9)
        assert 0.0 <= gate_features["novelty"] <= 1.0
        if gate_features["reward"]:
            assert record["reward_annotation"] is not None
            assert "postgres" in record["reward_annotation"].lower()
        if gate_features["pin"]:
            assert record["pin_annotation"]

    novelty_progression = [
        r["novelty_counters"]["config_index"] for r in records if r["gate_features"]["pin"]
    ]
    assert novelty_progression == sorted(novelty_progression)
