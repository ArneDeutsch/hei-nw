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
