import faiss
import numpy as np
import pytest

from hei_nw.store import ANNIndex


def test_ann_returns_expected_neighbor() -> None:
    x = np.eye(4, dtype="float32")
    idx = ANNIndex(dim=4)
    idx.add(x, [{"id": i} for i in range(4)])
    result = idx.search(x[0:1], k=1)
    assert result[0]["id"] == 0


def test_ann_meta_alignment() -> None:
    x = np.eye(4, dtype="float32")
    idx = ANNIndex(dim=4)
    with pytest.raises(ValueError):
        idx.add(x, [{"id": 0}])


def test_ann_backend_is_hnsw() -> None:
    idx = ANNIndex(dim=8)
    assert isinstance(idx.index, faiss.IndexHNSWFlat)
    assert idx.ef_search == 64


def test_ann_search_respects_ef_bounds() -> None:
    x = np.eye(4, dtype="float32")
    idx = ANNIndex(dim=4, ef_search=2)
    idx.add(x, [{"id": i} for i in range(4)])
    with pytest.raises(ValueError):
        idx.search(x[0:1], k=3)


def test_ann_recall_improves_with_higher_ef_search() -> None:
    state = np.random.get_state()
    np.random.seed(0)
    try:
        cluster_a = np.random.normal(0, 1, size=(200, 32)).astype("float32")
        cluster_b = np.random.normal(8, 0.1, size=(10, 32)).astype("float32")
    finally:
        np.random.set_state(state)
    all_vecs = np.vstack([cluster_a, cluster_b])
    meta = [{"id": i} for i in range(len(all_vecs))]
    index = ANNIndex(dim=32, m=4, ef_construction=50, ef_search=1)
    index.add(all_vecs, meta)

    target_id = cluster_a.shape[0]
    low = index.search(cluster_b[0:1], k=1)[0]["id"]
    index.set_ef_search(64)
    high = index.search(cluster_b[0:1], k=1)[0]["id"]

    assert low != target_id
    assert high == target_id
    assert high != low
