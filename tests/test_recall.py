from hei_nw.datasets import scenario_a
from hei_nw.recall import RecallService
from hei_nw.testing import DummyTokenizer


def test_recall_returns_token_ids_length_bound() -> None:
    tokenizer = DummyTokenizer()
    records = scenario_a.generate(n=5, seed=0)
    service = RecallService.build(records, tokenizer, max_mem_tokens=64)
    cue = records[0]["cues"][0]
    tokens = service.recall(cue)
    assert tokens, "recall should produce some token ids"
    assert len(tokens) <= 128
