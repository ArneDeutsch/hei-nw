import math
import random

import pytest

from hei_nw.gate import (
    NeuromodulatedGate,
    SalienceFeatures,
    bool_to_signal,
    novelty_from_similarity,
    surprise_from_logits,
    surprise_from_prob,
)


def test_gate_computes_weighted_salience() -> None:
    features = SalienceFeatures(surprise=0.8, novelty=0.6, reward=True, pin=False)
    gate = NeuromodulatedGate(alpha=1.0, beta=1.0, gamma=0.5, delta=0.8, threshold=1.0)
    score = gate.score(features)
    expected = 0.8 + 0.6 + 0.5 + 0.0
    assert math.isclose(score, expected, rel_tol=1e-6)
    decision = gate.decision(features)
    assert decision.should_write is True
    assert math.isclose(decision.contributions["surprise"], 0.8, rel_tol=1e-6)
    assert math.isclose(decision.contributions["novelty"], 0.6, rel_tol=1e-6)
    assert math.isclose(decision.contributions["reward"], 0.5, rel_tol=1e-6)
    assert math.isclose(decision.contributions["pin"], 0.0, rel_tol=1e-6)


def test_gate_threshold_controls_write_rate() -> None:
    rng = random.Random(0)
    features = [
        SalienceFeatures(
            surprise=rng.uniform(0.0, 3.0),
            novelty=rng.uniform(0.0, 1.0),
            reward=rng.random() < 0.05,
            pin=rng.random() < 0.01,
        )
        for _ in range(1024)
    ]
    permissive_gate = NeuromodulatedGate(threshold=1.8)
    strict_gate = NeuromodulatedGate(threshold=3.5)
    permissive_rate = sum(permissive_gate.should_write(f) for f in features) / len(features)
    strict_rate = sum(strict_gate.should_write(f) for f in features) / len(features)
    assert permissive_rate > strict_rate
    assert 0.4 < permissive_rate < 0.7
    assert strict_rate < 0.1


@pytest.mark.parametrize(
    "prob, expected",
    [
        (1.0, 0.0),
        (0.5, math.log(2)),
        (1e-4, pytest.approx(9.210, rel=1e-3)),
    ],
)
def test_surprise_from_prob(prob: float, expected: float) -> None:
    surprise = surprise_from_prob(prob)
    assert surprise == pytest.approx(expected, rel=1e-6)


def test_surprise_from_logits_matches_prob() -> None:
    logits = [2.0, 0.0, -1.0]
    target_index = 0
    surprise_logits = surprise_from_logits(logits, target_index)
    exp_logits = math.exp(2.0) / (math.exp(2.0) + math.exp(0.0) + math.exp(-1.0))
    surprise_probs = surprise_from_prob(exp_logits)
    assert surprise_logits == pytest.approx(surprise_probs, rel=1e-6)


@pytest.mark.parametrize(
    "similarity, expected",
    [
        (None, 1.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (1.0, 0.0),
    ],
)
def test_novelty_from_similarity(similarity: float | None, expected: float) -> None:
    novelty = novelty_from_similarity(similarity)
    assert novelty == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("flag", [True, False])
def test_bool_to_signal(flag: bool) -> None:
    signal = bool_to_signal(flag)
    assert signal in {0.0, 1.0}
    assert signal == float(flag)
