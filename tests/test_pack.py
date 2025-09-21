from pathlib import Path

import pytest

from transformers import AutoTokenizer

from hei_nw.pack import pack_trace, truncate_memory_tokens

TINY_MODEL = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2"


def test_pack_is_deterministic_and_capped() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_MODEL))  # type: ignore[no-untyped-call]
    trace = {
        "who": "Dana",
        "what": "backpack",
        "where": "Café Lumen",
        "when": "2025-09-10",
    }
    full = pack_trace(trace, tokenizer, 100)
    capped = pack_trace(trace, tokenizer, 5)
    assert capped == full[:5]
    assert pack_trace(trace, tokenizer, 5) == capped


def test_pack_handles_missing_fields() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_MODEL))  # type: ignore[no-untyped-call]
    trace = {"what": "backpack"}
    tokens = pack_trace(trace, tokenizer, 50)
    expected_text = "who: what: backpack where: when:"
    expected_ids = tokenizer(expected_text)["input_ids"][:50]
    assert tokens == expected_ids


def test_total_memory_token_cap_enforced() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_MODEL))  # type: ignore[no-untyped-call]
    trace = {
        "who": "Dana",
        "what": "backpack",
        "where": "Café Lumen",
        "when": "2025-09-10",
    }
    per_trace_tokens = pack_trace(trace, tokenizer, 32)
    combined = per_trace_tokens + per_trace_tokens
    capped = truncate_memory_tokens(combined, 48)
    assert capped == combined[:48]
    assert len(capped) == min(48, len(combined))


def test_pack_rejects_raw_text_payload() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_MODEL))  # type: ignore[no-untyped-call]
    trace = {"episode_text": "should not be packed"}
    with pytest.raises(ValueError):
        pack_trace(trace, tokenizer, 32)
