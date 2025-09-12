import torch

from hei_nw.keyer import DGKeyer, to_dense


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
