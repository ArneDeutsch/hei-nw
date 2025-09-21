from typing import cast

import pytest

from hei_nw.models.base import generate, load_base
from hei_nw.testing import DUMMY_MODEL_ID


def test_tokenizer_roundtrip_small() -> None:
    tok, _, _ = load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
    text = "Hello world"
    ids = tok.encode(text)  # type: ignore[attr-defined]
    decoded = tok.decode(ids, skip_special_tokens=True)  # type: ignore[attr-defined]
    assert decoded.strip() == text


@pytest.mark.slow
def test_generate_count_tokens_smoke() -> None:
    tok, _, _ = load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
    prompt = "Hello"
    out = generate(prompt, max_new_tokens=8)
    prompt_tokens = cast(int, out["prompt_tokens"])
    generated_tokens = cast(int, out["generated_tokens"])
    assert isinstance(out["text"], str)
    assert prompt_tokens == len(tok.encode(prompt))  # type: ignore[attr-defined]
    assert 0 < generated_tokens <= 8


def test_generate_signature_accepts_mem_tokens() -> None:
    load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
    a = generate("Hello", max_new_tokens=4)
    b = generate("Hello", max_new_tokens=4, mem_tokens=None)
    c = generate("Hello", max_new_tokens=4, mem_tokens=[])
    assert a["text"] == b["text"] == c["text"]
