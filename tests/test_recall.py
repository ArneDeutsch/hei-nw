from pathlib import Path

from transformers import AutoTokenizer

from hei_nw.datasets import scenario_a
from hei_nw.recall import RecallService

TINY_MODEL = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2"


def test_recall_returns_token_ids_length_bound() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_MODEL))  # type: ignore[no-untyped-call]
    records = scenario_a.generate(n=5, seed=0)
    service = RecallService.build(records, tokenizer, max_mem_tokens=64)
    cue = records[0]["cues"][0]
    tokens = service.recall(cue)
    assert tokens, "recall should produce some token ids"
    assert len(tokens) <= 128
