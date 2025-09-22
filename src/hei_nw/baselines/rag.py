"""Retrieval-augmented generation baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import faiss
import numpy as np

from hei_nw.metrics import (
    ComputeRecord,
    estimate_attention_flops,
    estimate_kv_bytes,
    recall_at_k,
)


class Embedder(Protocol):
    """Protocol for text embedders."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - Protocol
        """Return embeddings for *texts* as a 2D float32 array."""


class HFEmbedder:
    """Hugging Face encoder model wrapper."""

    def __init__(self, model_id: str = "intfloat/e5-small-v2", device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)  # type: ignore[no-untyped-call]
        self.model = AutoModel.from_pretrained(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from torch import Tensor, no_grad

        inputs = self.tok(list(texts), padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with no_grad():
            hidden: Tensor = self.model(**inputs).last_hidden_state.mean(dim=1)
        vecs = hidden.cpu().numpy().astype("float32")
        faiss.normalize_L2(vecs)
        return vecs


class FaissIndex:
    """Lightweight FAISS index wrapper."""

    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.docs: list[str] = []

    def add(self, vectors: np.ndarray, docs: Sequence[str]) -> None:
        if vectors.shape[0] != len(docs):
            msg = "vectors/docs length mismatch"
            raise ValueError(msg)
        self.index.add(vectors)
        self.docs.extend(docs)

    def search(self, query: np.ndarray, k: int) -> list[str]:
        _scores, idxs = self.index.search(query, k)
        return [self.docs[i] for i in idxs[0] if i >= 0]


@dataclass
class GenerationConfig:
    """Simple generation parameter container."""

    max_new_tokens: int = 128
    temperature: float | None = None
    top_p: float | None = None
    do_sample: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.do_sample:
            kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature or 1.0,
                    "top_p": self.top_p or 1.0,
                }
            )
        else:
            kwargs["do_sample"] = False
        return kwargs


def _model_config(model: Any) -> tuple[int, int, int, str]:
    cfg = model.config
    layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))
    hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
    dtype_attr = getattr(cfg, "dtype", None)
    if dtype_attr is None:
        dtype_attr = vars(cfg).get("torch_dtype")
    dtype_str = str(dtype_attr).replace("torch.", "") if dtype_attr else "float16"
    return layers, hidden, heads, dtype_str


def run_rag(
    model: Any,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    embedder: Embedder | None = None,
    k: int = 5,
    gen_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run retrieval-augmented generation on *records*.

    Each record should provide ``"documents"``, ``"query"``, ``"answers"``, and
    optional ``"expected"`` fields. Retrieved documents are concatenated to the
    query as context before generation.
    """

    emb = embedder or HFEmbedder()
    gen = GenerationConfig(**gen_cfg) if gen_cfg else GenerationConfig()
    layers, hidden, heads, dtype = _model_config(model)

    responses: list[dict[str, Any]] = []
    compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)

    for rec in records:
        docs = list(rec.get("documents", []))
        index: FaissIndex | None = None
        if docs:
            doc_vecs = emb.embed(docs)
            index = FaissIndex(doc_vecs.shape[1])
            index.add(doc_vecs, docs)

        query_vec = emb.embed([rec.get("query", "")])
        retrieved: list[str] = []
        if index is not None:
            retrieved = index.search(query_vec, k)
        recall = recall_at_k(retrieved, list(rec.get("answers", [])), k)

        context = "\n".join(retrieved)
        parts = [context, rec.get("query", "")]
        prompt = "\n".join(p for p in parts if p).strip()
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_toks = int(inputs["input_ids"].shape[-1])

        output = model.generate(**inputs, pad_token_id=tok.pad_token_id, **gen.to_kwargs())
        gen_ids = output[0][prompt_toks:]
        gen_toks = int(gen_ids.shape[0])
        text = tok.decode(gen_ids, skip_special_tokens=True)

        responses.append(
            {
                "text": text,
                "prompt_tokens": prompt_toks,
                "generated_tokens": gen_toks,
                "expected": rec.get("expected"),
                "retrieved": retrieved,
                "recall_at_k": recall,
            }
        )

        compute.attention_flops = (compute.attention_flops or 0) + estimate_attention_flops(
            prompt_toks, gen_toks, layers, hidden, heads
        )
        compute.kv_cache_bytes = (compute.kv_cache_bytes or 0) + estimate_kv_bytes(
            prompt_toks + gen_toks, hidden, dtype
        )

    return {"responses": responses, "compute": compute}
