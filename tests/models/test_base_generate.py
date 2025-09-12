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
    torch.manual_seed(3)
    adapter = EpisodicAdapter(hidden_size=2, n_heads=2)
    baseline = generate("Hello", max_new_tokens=1)
    with_mem = generate("Hello", max_new_tokens=1, adapter=adapter, mem_tokens=[2])
    assert baseline["text"] == " stairs"
    assert with_mem["text"] == " factors"
