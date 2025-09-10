from pathlib import Path
from typing import cast

import pytest

from hei_nw.models.base import generate, load_base

TINY_MODEL = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2"


def test_tokenizer_roundtrip_small() -> None:
    tok, _, _ = load_base(model_id=str(TINY_MODEL), quant_4bit=False)
    text = "Hello world"
    ids = tok.encode(text)  # type: ignore[attr-defined]
    decoded = tok.decode(ids, skip_special_tokens=True)  # type: ignore[attr-defined]
    assert decoded.strip() == text


@pytest.mark.slow
def test_generate_count_tokens_smoke() -> None:
    tok, _, _ = load_base(model_id=str(TINY_MODEL), quant_4bit=False)
    prompt = "Hello"
    out = generate(prompt, max_new_tokens=8)
    prompt_tokens = cast(int, out["prompt_tokens"])
    generated_tokens = cast(int, out["generated_tokens"])
    assert isinstance(out["text"], str)
    assert prompt_tokens == len(tok.encode(prompt))  # type: ignore[attr-defined]
    assert 0 < generated_tokens <= 8
