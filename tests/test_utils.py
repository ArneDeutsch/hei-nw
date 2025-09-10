import random

import numpy as np
import pytest

from hei_nw.utils.seed import set_global_seed


def test_seed_determinism() -> None:
    torch = pytest.importorskip("torch")

    set_global_seed(123)
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1)

    set_global_seed(123)
    assert random.random() == r1
    assert np.random.rand() == n1
    assert torch.allclose(torch.rand(1), t1)
