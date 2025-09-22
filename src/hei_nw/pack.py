"""Utilities for packing episodic traces into token IDs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_POINTER_KEYS = {"doc", "start", "end"}
_BANNED_TEXT_KEYS = {"episode_text", "raw_text", "snippet", "full_text", "text"}


def _validate_pointer_payload(trace: Mapping[str, Any]) -> None:
    for banned in _BANNED_TEXT_KEYS:
        if banned in trace:
            raise ValueError(f"trace contains disallowed key '{banned}'")
    pointer = trace.get("tokens_span_ref")
    if pointer is None:
        return
    if not isinstance(pointer, Mapping):
        raise TypeError("tokens_span_ref must be a mapping when provided")
    unknown = set(pointer.keys()) - _POINTER_KEYS
    if unknown:
        msg = f"tokens_span_ref contains unsupported fields: {sorted(unknown)}"
        raise ValueError(msg)
    doc = str(pointer.get("doc", "")).strip()
    if not doc:
        raise ValueError("tokens_span_ref.doc must be a non-empty string")
    start_raw = pointer.get("start")
    end_raw = pointer.get("end")
    if start_raw is None or end_raw is None:
        raise ValueError("tokens_span_ref start/end must be provided")
    try:
        start = int(start_raw)
        end = int(end_raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("tokens_span_ref start/end must be integers") from exc
    if start < 0 or end < 0:
        raise ValueError("tokens_span_ref offsets must be non-negative")
    if end <= start:
        raise ValueError("tokens_span_ref.end must be greater than start")


def _normalise_entity_slots(trace: Mapping[str, Any]) -> dict[str, str | dict[str, str]]:
    slots_source = trace.get("entity_slots")
    if isinstance(slots_source, Mapping):
        source = slots_source
    else:
        source = trace
    for banned in _BANNED_TEXT_KEYS:
        if banned in source:
            raise ValueError(f"entity slots may not contain disallowed key '{banned}'")
    slots: dict[str, str | dict[str, str]] = {
        key: str(source.get(key, "") or "").strip()
        for key in ("who", "what", "where", "when")
    }
    extras = source.get("extras")
    if extras is not None:
        if not isinstance(extras, Mapping):
            raise TypeError("entity slot extras must be a mapping if provided")
        slots["extras"] = {str(k): str(v) for k, v in extras.items()}
    return slots


def pack_trace(trace: Mapping[str, Any], tokenizer: Any, max_mem_tokens: int) -> list[int]:
    """Pack an episode trace into a list of token IDs.

    The packing follows a deterministic template with a stable field order and
    truncates the resulting token sequence to ``max_mem_tokens``. The function
    enforces pointer-only payloads: supplying ``tokens_span_ref`` requires a
    ``{doc,start,end}`` mapping and raw text fields such as ``episode_text`` are
    rejected.

    Parameters
    ----------
    trace:
        Mapping describing the episode. It may contain a nested
        ``entity_slots`` mapping with ``who``, ``what``, ``where`` and ``when``
        fields or expose those fields directly. Raw text payloads are not
        permitted; callers must provide pointer metadata instead.
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

    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping")
    _validate_pointer_payload(trace)
    slots = _normalise_entity_slots(trace)
    template = " ".join(
        [
            f"who: {slots['who']}".strip(),
            f"what: {slots['what']}".strip(),
            f"where: {slots['where']}".strip(),
            f"when: {slots['when']}".strip(),
        ]
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
