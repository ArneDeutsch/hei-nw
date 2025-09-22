import pytest

from hei_nw.pack import _normalise_entity_slots, pack_trace, truncate_memory_tokens
from hei_nw.testing import DummyTokenizer


def test_pack_is_deterministic_and_capped() -> None:
    tokenizer = DummyTokenizer()
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
    tokenizer = DummyTokenizer()
    trace = {"what": "backpack"}
    tokens = pack_trace(trace, tokenizer, 50)
    expected_text = "who: what: backpack where: when:"
    expected_ids = tokenizer(expected_text)["input_ids"][:50]
    assert tokens == expected_ids


def test_total_memory_token_cap_enforced() -> None:
    tokenizer = DummyTokenizer()
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
    tokenizer = DummyTokenizer()
    trace = {"episode_text": "should not be packed"}
    with pytest.raises(ValueError):
        pack_trace(trace, tokenizer, 32)


def test_pack_requires_mapping_trace() -> None:
    tokenizer = DummyTokenizer()
    with pytest.raises(TypeError, match="trace must be a mapping"):
        pack_trace(["not", "a", "mapping"], tokenizer, 8)


def test_pack_rejects_non_mapping_pointer_payload() -> None:
    tokenizer = DummyTokenizer()
    trace = {"tokens_span_ref": "bad"}
    with pytest.raises(TypeError, match="tokens_span_ref must be a mapping"):
        pack_trace(trace, tokenizer, 16)


@pytest.mark.parametrize(
    "pointer, expected_message",
    [
        ({"doc": "", "start": 0, "end": 1}, "doc must be a non-empty string"),
        ({"doc": "Story", "end": 2}, "start/end must be provided"),
        ({"doc": "Story", "start": "x", "end": 2}, "must be integers"),
        ({"doc": "Story", "start": -1, "end": 2}, "non-negative"),
        ({"doc": "Story", "start": 2, "end": 2}, "must be greater than start"),
    ],
)
def test_pack_validates_pointer_offsets(pointer: dict[str, object], expected_message: str) -> None:
    tokenizer = DummyTokenizer()
    trace = {"tokens_span_ref": pointer}
    with pytest.raises(ValueError, match=expected_message):
        pack_trace(trace, tokenizer, 16)


def test_pack_rejects_unknown_pointer_fields() -> None:
    tokenizer = DummyTokenizer()
    trace = {
        "tokens_span_ref": {"doc": "Story", "start": 1, "end": 2, "extra": True},
    }
    with pytest.raises(ValueError, match="unsupported fields"):
        pack_trace(trace, tokenizer, 16)


def test_pack_allows_valid_pointer_payload() -> None:
    tokenizer = DummyTokenizer()
    trace = {
        "who": "Kim",
        "tokens_span_ref": {"doc": "Story", "start": 1, "end": 3},
    }
    tokens = pack_trace(trace, tokenizer, 10)
    expected = tokenizer("who: Kim what: where: when:")["input_ids"][:10]
    assert tokens == expected


def test_pack_uses_nested_entity_slots() -> None:
    tokenizer = DummyTokenizer()
    trace = {
        "who": "outer",  # ignored because entity_slots takes precedence
        "entity_slots": {
            "who": "  inner  ",
            "what": "bag",
            "where": "park",
            "when": "today",
        },
    }
    tokens = pack_trace(trace, tokenizer, 16)
    expected = tokenizer("who: inner what: bag where: park when: today")["input_ids"][:16]
    assert tokens == expected


def test_pack_rejects_non_mapping_entity_extras() -> None:
    tokenizer = DummyTokenizer()
    trace = {"entity_slots": {"extras": ["oops"]}}
    with pytest.raises(TypeError, match="extras must be a mapping"):
        pack_trace(trace, tokenizer, 16)


def test_pack_rejects_entity_slots_with_text_payload() -> None:
    tokenizer = DummyTokenizer()
    trace = {"entity_slots": {"who": "A", "episode_text": "not allowed"}}
    with pytest.raises(ValueError, match="disallowed key"):
        pack_trace(trace, tokenizer, 16)


def test_normalise_entity_slots_stringifies_extras() -> None:
    slots = _normalise_entity_slots({"entity_slots": {"extras": {1: 2}}})
    assert slots["extras"] == {"1": "2"}


def test_truncate_memory_tokens_requires_positive_cap() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        truncate_memory_tokens([1, 2, 3], 0)
