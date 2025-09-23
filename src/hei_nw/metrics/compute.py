"""Compute cost estimation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

_DTYPE_SIZES = {
    "fp8": 1,
    "int8": 1,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
    "float64": 8,
    "fp64": 8,
}


def _dtype_nbytes(dtype: str) -> int:
    try:
        return _DTYPE_SIZES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def estimate_attention_flops(
    prompt_toks: int,
    gen_toks: int,
    n_layers: int,
    d_model: int,
    heads: int,
) -> int:
    """Estimate self-attention FLOPs for decoder-only generation.

    The calculation is a coarse big-O style estimate assuming full attention.
    It scales with the square of the total sequence length and linearly with
    the model width and number of layers.
    """

    seq_len = prompt_toks + gen_toks
    d_head = d_model / max(heads, 1)
    flops_per_layer = 2 * heads * seq_len * seq_len * d_head
    return int(n_layers * flops_per_layer)


def estimate_kv_bytes(tokens: int, d_model: int, dtype: str = "fp16") -> int:
    """Estimate bytes required for key/value caches per layer.

    Each token stores a key and value vector of size ``d_model`` in the given
    ``dtype``. This function returns the total bytes for both caches across all
    tokens for a single layer.
    """

    nbytes = _dtype_nbytes(dtype)
    return tokens * d_model * nbytes * 2


@dataclass
class ComputeRecord:
    """Record of compute-related metrics.

    Fields are optional but always included in serialized output.
    """

    attention_flops: int | None = None
    kv_cache_bytes: int | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None

    def model_dump(self) -> dict[str, int | None]:
        """Return a dictionary with all fields present."""

        return asdict(self)
