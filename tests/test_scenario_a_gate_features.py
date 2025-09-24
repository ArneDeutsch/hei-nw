"""Scenario A gate feature heuristics tests."""

from __future__ import annotations

import math
from collections.abc import Mapping

from hei_nw.datasets.scenario_a import generate


def test_all_records_have_gate_features() -> None:
    records = generate(6, seed=5)
    assert records

    for record in records:
        gate_features = record["gate_features"]
        assert isinstance(gate_features, Mapping)
        for key in ("surprise", "novelty", "reward", "pin"):
            assert key in gate_features
            assert gate_features[key] is not None


def test_feature_ranges() -> None:
    records = generate(12, seed=7)
    positives = [r for r in records if r["should_remember"]]
    negatives = [r for r in records if not r["should_remember"]]
    assert positives
    assert negatives

    pos_surprise = []
    neg_surprise = []
    pos_novelty = []
    neg_novelty = []

    for record in records:
        gate_features = record["gate_features"]
        surprise = gate_features["surprise"]
        novelty = gate_features["novelty"]
        reward = gate_features["reward"]
        pin = gate_features["pin"]

        assert isinstance(gate_features, Mapping)
        assert math.isfinite(float(surprise))
        assert surprise >= 0.0
        assert 0.0 <= float(novelty) <= 1.0
        assert isinstance(reward, bool)
        assert isinstance(pin, bool)

        if record["should_remember"]:
            pos_surprise.append(float(surprise))
            pos_novelty.append(float(novelty))
        else:
            neg_surprise.append(float(surprise))
            neg_novelty.append(float(novelty))

    assert pos_surprise and neg_surprise
    assert pos_novelty and neg_novelty

    avg_pos_surprise = sum(pos_surprise) / len(pos_surprise)
    avg_neg_surprise = sum(neg_surprise) / len(neg_surprise)
    avg_pos_novelty = sum(pos_novelty) / len(pos_novelty)
    avg_neg_novelty = sum(neg_novelty) / len(neg_novelty)

    assert avg_pos_surprise > avg_neg_surprise
    assert avg_pos_novelty > avg_neg_novelty
