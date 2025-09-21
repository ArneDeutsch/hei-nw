import math
from collections.abc import Sequence
from typing import cast

from hei_nw.datasets.scenario_a import generate


def test_shapes_and_fields() -> None:
    records = generate(3, seed=0)
    assert len(records) == 6
    first = records[0]
    expected = {
        "episode_text",
        "cues",
        "answers",
        "should_remember",
        "lag",
        "group_id",
        "gate_features",
    }
    assert expected <= first.keys()
    assert isinstance(first["episode_text"], str)
    assert isinstance(first["cues"], list)
    assert isinstance(first["answers"], list)
    assert len(first["cues"]) == len(first["answers"])
    assert isinstance(first["should_remember"], bool)
    assert isinstance(first["lag"], int)
    assert isinstance(first["group_id"], int)
    gate_features = first["gate_features"]
    assert {"surprise", "novelty", "reward", "pin"} <= gate_features.keys()
    assert 0.0 <= gate_features["novelty"] <= 1.0
    assert gate_features["surprise"] >= 0.0


def test_hard_negative_rate() -> None:
    records = generate(20, seed=1)
    neg = sum(1 for r in records if not r["should_remember"])
    rate = neg / len(records)
    assert math.isclose(rate, 0.5, rel_tol=0.2)


def test_lag_bins_cover() -> None:
    lag_spec: dict[str, Sequence[int]] = {"bins": [0, 1, 3, 7, 30]}
    records = generate(20, seed=2, lag_spec=lag_spec)
    bins = lag_spec["bins"]

    def bin_index(lag: int) -> int | None:
        for i in range(len(bins) - 1):
            if bins[i] <= lag < bins[i + 1]:
                return i
        return None

    covered = {bin_index(cast(int, r["lag"])) for r in records}
    assert covered == set(range(len(bins) - 1))
