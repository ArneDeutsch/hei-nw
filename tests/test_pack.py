from pathlib import Path

from transformers import AutoTokenizer

from hei_nw.pack import pack_trace

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
    expected_text = "<episodic>\n" "who:\n" "what:backpack\n" "where:\n" "when:\n" "</episodic>"
    expected_ids = tokenizer(expected_text)["input_ids"][:50]
    assert tokens == expected_ids
