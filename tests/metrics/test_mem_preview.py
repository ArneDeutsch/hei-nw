from pathlib import Path

from transformers import AutoTokenizer

from hei_nw.eval.harness import _decode_mem_preview
from hei_nw.pack import pack_trace

TINY_MODEL = Path(__file__).resolve().parents[2] / "models" / "tiny-gpt2"


def test_mem_preview_has_no_global_prefix_tokens() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_MODEL))  # type: ignore[no-untyped-call]
    trace = {
        "who": "Dana",
        "what": "backpack",
        "where": "Café Lumen",
        "when": "2025-09-10",
    }
    token_ids = pack_trace(trace, tokenizer, 32)
    preview = _decode_mem_preview(tokenizer, token_ids[:8])

    assert preview, "memory preview should include decoded tokens"
    first_token = preview[0].lstrip("Ġ")
    assert first_token.startswith("who"), f"unexpected first token: {preview[0]}"
    assert all("<episodic>" not in token for token in preview)
    assert all("•" not in token for token in preview)
