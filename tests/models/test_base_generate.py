from pathlib import Path

import pytest

from hei_nw.adapter import EpisodicAdapter
from hei_nw.models.base import generate, load_base

TINY_MODEL = Path(__file__).resolve().parents[2] / "models" / "tiny-gpt2"


def setup_module() -> None:
    load_base(model_id=str(TINY_MODEL), quant_4bit=False)


def test_generate_stops_on_substring() -> None:
    out = generate("Hello", max_new_tokens=8, stop="stairs")
    assert "stairs" not in out["text"]
    assert out["generated_tokens"] < 8


def test_warning_when_adapter_and_mem_tokens() -> None:
    adapter = EpisodicAdapter(hidden_size=1, n_heads=1)
    with pytest.warns(UserWarning):
        generate("Hello", max_new_tokens=1, adapter=adapter, mem_tokens=[0])
