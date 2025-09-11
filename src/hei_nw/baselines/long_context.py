"""Long-context baseline module."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hei_nw.metrics import (
    ComputeRecord,
    estimate_attention_flops,
    estimate_kv_bytes,
)


@dataclass
class GenerationConfig:
    """Simple container for generation parameters."""

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


def build_context(record: Mapping[str, str]) -> str:
    """Pack long-context prompt from a record.

    Parameters
    ----------
    record:
        Mapping containing ``"context"`` and ``"query"`` fields.

    Returns
    -------
    str
        Combined prompt string.
    """
    context = record.get("context", "")
    query = record.get("query", "")
    parts = [context.strip(), query.strip()]
    return "\n".join(p for p in parts if p)


def _model_config(model: Any) -> tuple[int, int, int, str]:
    """Extract basic config fields from a model."""
    cfg = model.config
    layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))
    hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
    dtype_attr = getattr(cfg, "dtype", None)
    if dtype_attr is None:
        dtype_attr = vars(cfg).get("torch_dtype")
    dtype_str = str(dtype_attr).replace("torch.", "") if dtype_attr else "float16"
    return layers, hidden, heads, dtype_str


def run_long_context(
    model: Any,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    gen_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the long-context baseline on a set of records.

    Parameters
    ----------
    model:
        Hugging Face model used for generation.
    tok:
        Tokenizer paired with the model.
    records:
        Iterable of mappings with ``context`` and ``query``.
    gen_cfg:
        Optional generation configuration overrides.

    Returns
    -------
    dict
        ``{"responses": [...], "compute": ComputeRecord}``
    """
    gen = GenerationConfig(**gen_cfg) if gen_cfg else GenerationConfig()
    layers, hidden, heads, dtype = _model_config(model)

    responses: list[dict[str, Any]] = []
    compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)

    for rec in records:
        prompt = build_context(rec)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_toks = int(inputs["input_ids"].shape[-1])

        output = model.generate(**inputs, **gen.to_kwargs())
        gen_ids = output[0][prompt_toks:]
        gen_toks = int(gen_ids.shape[0])
        text = tok.decode(gen_ids, skip_special_tokens=True)

        responses.append(
            {
                "text": text,
                "prompt_tokens": prompt_toks,
                "generated_tokens": gen_toks,
                "expected": rec.get("expected"),
            }
        )

        compute.attention_flops = (compute.attention_flops or 0) + estimate_attention_flops(
            prompt_toks, gen_toks, layers, hidden, heads
        )
        compute.kv_cache_bytes = (compute.kv_cache_bytes or 0) + estimate_kv_bytes(
            prompt_toks + gen_toks, hidden, dtype
        )

    return {"responses": responses, "compute": compute}
