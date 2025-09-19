import math

from hei_nw.datasets import scenario_a
from hei_nw.store import EpisodicStore


class DummyTokenizer:
    def tokenize(self, text: str) -> list[str]:  # pragma: no cover - trivial
        return text.split()


def test_build_and_query_top1_self() -> None:
    tok = DummyTokenizer()
    records = scenario_a.generate(n=2, seed=0)
    store = EpisodicStore.from_records(records, tok, max_mem_tokens=64)
    positive = next(r for r in records if r["should_remember"])
    out = store.query(
        positive["cues"][0],
        group_id=positive["group_id"],
        should_remember=positive["should_remember"],
    )
    assert out["selected"][0]["group_id"] == positive["group_id"]
    diagnostics = out["diagnostics"]
    assert diagnostics["pre_top1_group"] == positive["group_id"]
    assert diagnostics["post_top1_group"] == positive["group_id"]
    assert diagnostics["rank_delta"] == 0


def test_near_miss_and_collision_counters() -> None:
    tok = DummyTokenizer()
    records = scenario_a.generate(n=2, seed=0)
    store = EpisodicStore.from_records(records, tok, max_mem_tokens=64)
    near_miss_found = False
    for r in records:
        if r["should_remember"]:
            continue
        out_nm = store.query(r["cues"][0], group_id=r["group_id"], should_remember=False)
        if out_nm["diagnostics"]["near_miss"]:
            near_miss_found = True
            break
    assert near_miss_found
    pos0 = next(r for r in records if r["should_remember"])
    pos1 = next(r for r in records if r["should_remember"] and r["group_id"] != pos0["group_id"])
    out_col = store.query(pos0["cues"][0], group_id=pos1["group_id"], should_remember=True)
    diag = out_col["diagnostics"]
    assert diag["collision"] is True
    assert "pre_top1_group" in diag and "post_top1_group" in diag


def test_custom_hopfield_parameters() -> None:
    tok = DummyTokenizer()
    records = scenario_a.generate(n=2, seed=0)
    store = EpisodicStore.from_records(
        records,
        tok,
        max_mem_tokens=64,
        hopfield_steps=3,
        hopfield_temperature=0.5,
    )
    assert store.hopfield.steps == 3
    assert math.isclose(store.hopfield.temperature, 0.5)
