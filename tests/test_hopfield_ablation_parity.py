import numpy as np
import torch

from hei_nw.store import EpisodicStore, HopfieldReadout


class DummyKeyer:
    def __init__(self, dense_vector: torch.Tensor) -> None:
        self._dense = dense_vector

    def __call__(self, _: torch.Tensor) -> dict[str, torch.Tensor | int]:
        dim = int(self._dense.numel())
        indices = torch.arange(dim, dtype=torch.int64).unsqueeze(0)
        values = self._dense.unsqueeze(0)
        return {"indices": indices, "values": values, "dim": dim}


class DummyIndex:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results

    def search(self, query: np.ndarray, k: int) -> list[dict[str, object]]:  # noqa: ARG002
        return list(self._results[:k])


class FakeStore(EpisodicStore):
    def _embed(self, cue_text: str) -> torch.Tensor:  # noqa: D401
        """Return a placeholder embedding; actual vector is provided by the keyer."""

        return torch.zeros(1, 1, self._embed_dim, dtype=torch.float32)


def test_hopfield_ablation_changes_top_rank() -> None:
    query_vec = torch.tensor([0.9, 0.1], dtype=torch.float32)
    keyer = DummyKeyer(query_vec)
    candidate_vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.8, 0.6], dtype=np.float32),
        np.array([-0.2, 0.98], dtype=np.float32),
    ]
    ann_results = [
        {"trace": {"id": "baseline"}, "key_vector": candidate_vectors[0], "group_id": 1, "score": 0.99},
        {"trace": {"id": "hopfield"}, "key_vector": candidate_vectors[1], "group_id": 2, "score": 0.80},
        {"trace": {"id": "other"}, "key_vector": candidate_vectors[2], "group_id": 3, "score": 0.70},
    ]
    store = FakeStore(
        keyer=keyer,
        index=DummyIndex(ann_results),
        hopfield=HopfieldReadout(torch.eye(query_vec.numel()), steps=2),
        tokenizer=None,
        vectors=candidate_vectors,
        group_ids={1, 2, 3},
        embed_dim=query_vec.numel(),
        max_mem_tokens=0,
    )

    without_hopfield = store.query("cue", top_k_candidates=3, return_m=1, use_hopfield=False)
    with_hopfield = store.query("cue", top_k_candidates=3, return_m=1, use_hopfield=True)

    assert [trace["id"] for trace in without_hopfield["selected"]] == ["baseline"]
    assert [trace["id"] for trace in with_hopfield["selected"]] == ["hopfield"]
    assert [cand["trace"]["id"] for cand in without_hopfield["candidates"]] == ["baseline", "hopfield", "other"]
    assert [cand["trace"]["id"] for cand in with_hopfield["candidates"]] == ["baseline", "hopfield", "other"]
