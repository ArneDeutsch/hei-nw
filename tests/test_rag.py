from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest
import requests  # type: ignore[import-untyped]

from hei_nw.baselines.rag import run_rag
from hei_nw.models.base import load_base
from hei_nw.testing import DUMMY_MODEL_ID


class ToyEmbedder:
    """Deterministic embedder using hashed one-hot vectors."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16) % self.dim
            vec = np.zeros(self.dim, dtype="float32")
            vec[h] = 1.0
            vecs.append(vec)
        return np.stack(vecs)


def test_index_and_query_toyembedder() -> None:
    tok, model, _ = load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
    records = [
        {
            "documents": ["alpha", "beta", "gamma"],
            "query": "beta",
            "answers": ["beta"],
            "expected": "beta",
        }
    ]
    out = run_rag(model, tok, records, embedder=ToyEmbedder(), k=1, gen_cfg={"max_new_tokens": 1})
    resp = out["responses"][0]
    assert resp["recall_at_k"] == 1.0
    compute = out["compute"].model_dump()
    assert compute["attention_flops"] is not None
    assert compute["kv_cache_bytes"] is not None


@pytest.mark.slow
def test_hfembedder_smoke() -> None:
    from hei_nw.baselines.rag import HFEmbedder

    try:
        emb = HFEmbedder()
        vecs = emb.embed(["hello", "world"])
    except requests.exceptions.RequestException:
        pytest.skip("HF embedding model not available")
    assert vecs.shape[0] == 2
