import math

import numpy as np
import torch

from hei_nw.datasets import scenario_a
from hei_nw.store import EpisodicStore, HopfieldReadout


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
    baseline_candidates = out.get("baseline_candidates")
    assert baseline_candidates is not None
    assert baseline_candidates[0]["group_id"] == diagnostics["pre_top1_group"]


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
            assert "baseline_diagnostics" in out_nm
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


def test_hopfield_refinement_promotes_better_candidate() -> None:
    dense_values = torch.tensor([[0.4, 0.6]], dtype=torch.float32)

    class StubKeyer:
        def __init__(self) -> None:
            self.d = dense_values.shape[-1]
            self.k = dense_values.shape[-1]
            self._indices = torch.arange(self.k).unsqueeze(0)

        def __call__(self, _H: torch.Tensor) -> dict[str, torch.Tensor | int]:
            return {
                "indices": self._indices,
                "values": dense_values.clone(),
                "dim": self.d,
            }

    class StubIndex:
        def __init__(self, results: list[dict[str, object]]) -> None:
            self._results = results
            self.ef_search = len(results)

        def search(self, _dense: np.ndarray, k: int) -> list[dict[str, object]]:
            return self._results[:k]

    patterns = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    store = EpisodicStore.__new__(EpisodicStore)
    store.keyer = StubKeyer()
    store.hopfield = HopfieldReadout(patterns, steps=1, temperature=0.5)
    store._vectors = []
    store._group_ids = {1, 2}
    store._embed_dim = dense_values.shape[-1]
    store.max_mem_tokens = 8
    store._hopfield_blend = 0.2
    store._hopfield_margin = 0.0
    store.tokenizer = None

    def fake_hash_embed(_text: str, _tokenizer: object, _dim: int) -> torch.Tensor:
        return dense_values.view(1, 1, -1)

    store._hash_embed = fake_hash_embed  # type: ignore[assignment]

    wrong_vec = np.array([0.6, 0.4], dtype="float32")
    correct_vec = np.array([0.3, 0.7], dtype="float32")
    results = [
        {"group_id": 1, "score": 0.401, "key_vector": wrong_vec, "trace": {"group_id": 1}},
        {"group_id": 2, "score": 0.400, "key_vector": correct_vec, "trace": {"group_id": 2}},
    ]
    store.index = StubIndex(results)

    baseline = store.query("cue", use_hopfield=False, return_m=1)
    assert baseline["candidates"][0]["group_id"] == 1

    refined = store.query("cue", use_hopfield=True, return_m=1, group_id=2, should_remember=True)
    assert refined["candidates"][0]["group_id"] == 2
    assert refined["diagnostics"]["rank_delta"] > 0
