"""Utilities for packing episodic traces into token IDs."""

from __future__ import annotations

from typing import Any, Sequence


def pack_trace(trace: dict[str, Any], tokenizer: Any, max_mem_tokens: int) -> list[int]:
    """Pack an episode trace into a list of token IDs.

    The packing follows a deterministic template with a stable field order and
    truncates the resulting token sequence to ``max_mem_tokens``.

    Parameters
    ----------
    trace:
        Mapping containing optional ``who``, ``what``, ``where`` and ``when``
        fields describing the episode. Missing fields are treated as empty
        strings. All values are converted to strings and stripped.
    tokenizer:
        HuggingFace-style tokenizer providing a ``__call__`` method that returns
        a mapping with an ``"input_ids"`` list.
    max_mem_tokens:
        Maximum number of tokens to return. The output is truncated to this
        length.

    Returns
    -------
    list[int]
        Token IDs representing the packed episode trace. Length is at most
        ``max_mem_tokens``.
    """

    fields = {key: str(trace.get(key, "")).strip() for key in ("who", "what", "where", "when")}
    template = (
        "<episodic>\n"
        f"who:{fields['who']}\n"
        f"what:{fields['what']}\n"
        f"where:{fields['where']}\n"
        f"when:{fields['when']}\n"
        "</episodic>"
    )
    input_ids: list[int] = tokenizer(template)["input_ids"]
    return input_ids[:max_mem_tokens]


def truncate_memory_tokens(tokens: Sequence[int], max_total_tokens: int) -> list[int]:
    """Return *tokens* truncated to at most ``max_total_tokens`` elements.

    Parameters
    ----------
    tokens:
        Sequence of token identifiers representing concatenated episodic
        memories.
    max_total_tokens:
        Maximum number of tokens to keep. Must be a positive integer.

    Returns
    -------
    list[int]
        Truncated token identifiers. The returned list is always a new list,
        ensuring callers can mutate it without affecting the input sequence.
    """

    if max_total_tokens <= 0:
        msg = "max_total_tokens must be a positive integer"
        raise ValueError(msg)
    return list(tokens[:max_total_tokens])
