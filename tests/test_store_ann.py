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
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype="float32",
    )
    meta = [{"id": i} for i in range(len(vectors))]
    index = ANNIndex(dim=2, m=2, ef_construction=10, ef_search=1)
    index.add(vectors, meta)

    query = np.array([[0.95, 0.05]], dtype="float32")
    low = index.search(query, k=1)[0]["id"]
    with pytest.raises(ValueError):
        index.search(query, k=2)

    index.set_ef_search(2)
    high = index.search(query, k=2)
    ids = {res["id"] for res in high}

    assert low in ids
    assert ids == {0, 1}


def test_ann_search_orders_by_descending_similarity() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.7, 0.7],
        ],
        dtype="float32",
    )
    meta = [{"id": i} for i in range(len(vectors))]
    index = ANNIndex(dim=2)
    index.add(vectors, meta)

    results = index.search(vectors[0:1], k=3)
    scores = [res["score"] for res in results]
    distances = [res["distance"] for res in results]

    assert scores == sorted(scores, reverse=True)
    assert distances == sorted(distances)
