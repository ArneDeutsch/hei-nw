from hei_nw.eval.harness import _decode_mem_preview
from hei_nw.pack import pack_trace
from hei_nw.testing import DummyTokenizer


def test_mem_preview_has_no_global_prefix_tokens() -> None:
    tokenizer = DummyTokenizer()
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
