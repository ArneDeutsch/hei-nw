from pathlib import Path

import torch

from hei_nw.adapter import EpisodicAdapter
from hei_nw.models.base import generate, load_base

TINY_MODEL = Path(__file__).resolve().parents[2] / "models" / "tiny-gpt2"


def setup_module() -> None:
    load_base(model_id=str(TINY_MODEL), quant_4bit=False)


def test_generate_stops_on_substring() -> None:
    out = generate("Hello", max_new_tokens=8, stop="stairs")
    assert "stairs" not in out["text"]
    assert out["generated_tokens"] < 8


def test_mem_tokens_affect_generation() -> None:
    torch.manual_seed(5)
    adapter = EpisodicAdapter(hidden_size=2, n_heads=2, scale=0.5)
    baseline = generate("Hello", max_new_tokens=1)
    with_mem = generate("Hello", max_new_tokens=1, adapter=adapter, mem_tokens=[1])
    assert baseline["text"].strip() == "stairs"
    assert with_mem["text"].strip() == "factors"


def test_adapter_scale_modulates_generation() -> None:
    torch.manual_seed(5)
    baseline = generate("Hello", max_new_tokens=1)

    adapter_zero = EpisodicAdapter(hidden_size=2, n_heads=2, scale=0.0)
    torch.manual_seed(5)
    zero_scale = generate(
        "Hello",
        max_new_tokens=1,
        adapter=adapter_zero,
        mem_tokens=[1],
    )

    adapter_strong = EpisodicAdapter(hidden_size=2, n_heads=2, scale=0.5)
    torch.manual_seed(5)
    strong_scale = generate(
        "Hello",
        max_new_tokens=1,
        adapter=adapter_strong,
        mem_tokens=[1],
    )

    assert zero_scale["text"].strip() == baseline["text"].strip()
    assert strong_scale["text"].strip() == "factors"
