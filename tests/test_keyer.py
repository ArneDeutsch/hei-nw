import numpy as np
import torch

from hei_nw.keyer import DGKeyer, to_dense
from hei_nw.store import EpisodicStore


def test_k_wta_sparsity_invariants() -> None:
    keyer = DGKeyer(d=32, k=4)
    H = torch.randn(5, 7, 8)
    key = keyer(H)
    dense = to_dense(key)
    # exactly k non-zero entries per sample
    assert (dense != 0).sum(dim=-1).eq(keyer.k).all()
    # L1 norm of retained values is 1
    l1 = key["values"].abs().sum(dim=-1)
    assert torch.allclose(l1, torch.ones_like(l1))
    # stable under positive scaling
    scaled = keyer(H * 2.0)
    assert torch.equal(key["indices"], scaled["indices"])
    assert torch.allclose(key["values"], scaled["values"])


def test_to_dense_roundtrip() -> None:
    keyer = DGKeyer(d=16, k=4)
    H = torch.randn(2, 3, 8)
    key = keyer(H)
    dense = to_dense(key)
    topk = dense.abs().topk(keyer.k, dim=-1)
    gathered = dense.gather(-1, key["indices"])
    assert torch.equal(topk.indices, key["indices"])
    assert torch.allclose(gathered, key["values"])
    assert dense.shape == (2, 16)


def test_cli_k_threads() -> None:
    class DummyTokenizer:
        def tokenize(self, text: str) -> list[str]:  # pragma: no cover - trivial helper
            return text.split()

    records = [
        {
            "episode_text": "alpha beta gamma delta",
            "answers": ["alpha", "beta", "gamma", "delta"],
            "group_id": 0,
            "should_remember": True,
        },
        {
            "episode_text": "epsilon zeta eta theta",
            "answers": ["epsilon", "zeta", "eta", "theta"],
            "group_id": 1,
            "should_remember": True,
        },
    ]
    tok = DummyTokenizer()
    store_k1 = EpisodicStore.from_records(
        records,
        tok,
        max_mem_tokens=16,
        embed_dim=8,
        keyer=DGKeyer(d=16, k=1),
    )
    store_k3 = EpisodicStore.from_records(
        records,
        tok,
        max_mem_tokens=16,
        embed_dim=8,
        keyer=DGKeyer(d=16, k=3),
    )
    nnz_k1 = {np.count_nonzero(vec) for vec in store_k1._vectors}
    nnz_k3 = {np.count_nonzero(vec) for vec in store_k3._vectors}
    assert nnz_k1 == {1}
    assert nnz_k3 == {3}
