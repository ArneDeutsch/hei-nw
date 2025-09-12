"""Recall service for episodic memory retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .pack import pack_trace
from .store import EpisodicStore

__all__ = ["RecallService"]


class RecallService:
    """Service wrapping :class:`EpisodicStore` to return memory tokens.

    The service first builds an :class:`EpisodicStore` from raw records and a
    tokenizer. A call to :meth:`recall` queries the store with a natural
    language cue and returns a concatenated list of token IDs representing
    retrieved episodic traces. The output is truncated to a maximum of
    ``128`` tokens, and each individual trace is capped by ``max_mem_tokens``.
    """

    def __init__(
        self,
        store: EpisodicStore,
        tokenizer: Any,
        max_mem_tokens: int,
        return_m: int = 4,
    ) -> None:
        self.store = store
        self.tokenizer = tokenizer
        self.max_mem_tokens = max_mem_tokens
        self.return_m = return_m

    @classmethod
    def build(
        cls,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_mem_tokens: int,
        return_m: int = 4,
    ) -> RecallService:
        """Construct a :class:`RecallService` from raw records."""

        store = EpisodicStore.from_records(records, tokenizer, max_mem_tokens)
        return cls(store, tokenizer, max_mem_tokens, return_m)

    def recall(self, cue_text: str) -> list[int]:
        """Return packed memory token IDs for *cue_text*.

        Parameters
        ----------
        cue_text:
            Natural-language description used to query the store.

        Returns
        -------
        list[int]
            Token IDs representing retrieved episodic traces. The list length
            is at most ``128`` tokens.
        """

        result = self.store.query(cue_text, return_m=self.return_m)
        tokens: list[int] = []
        for trace in result["selected"]:
            answers = trace.get("answers", [])
            fields = {
                key: answers[i] if i < len(answers) else ""
                for i, key in enumerate(["who", "what", "where", "when"])
            }
            tokens.extend(pack_trace(fields, self.tokenizer, self.max_mem_tokens))
            if len(tokens) >= 128:
                break
        return tokens[:128]
