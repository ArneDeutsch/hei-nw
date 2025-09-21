from collections.abc import Callable
from typing import Any

from hei_nw.datasets import scenario_b, scenario_c, scenario_d, scenario_e


def _generators() -> list[Callable[..., list[dict[str, Any]]]]:
    return [
        scenario_b.generate,
        scenario_c.generate,
        scenario_d.generate,
        scenario_e.generate,
    ]


def test_each_generator_min_sizes() -> None:
    for gen in _generators():
        records = gen(5, seed=0)
        assert len(records) >= 5
        first = records[0]
        assert {"context", "query", "expected", "should_remember"} <= first.keys()
        assert "gate_features" in first
        gate_features = first["gate_features"]
        assert {"surprise", "novelty", "reward", "pin"} <= gate_features.keys()


def test_should_remember_present() -> None:
    for gen in _generators():
        records = gen(6, seed=1)
        assert any(r["should_remember"] for r in records)
        assert any(not r["should_remember"] for r in records)
