"""Regression tests for newline stop handling in ``generate``."""

from __future__ import annotations

from pathlib import Path

import pytest

import hei_nw.models.base as base
from hei_nw.adapter import EpisodicAdapter

TINY_MODEL = Path(__file__).resolve().parents[2] / "models" / "tiny-gpt2"


def setup_module() -> None:
    base.load_base(model_id=str(TINY_MODEL), quant_4bit=False)


def _build_adapter() -> EpisodicAdapter:
    return EpisodicAdapter(hidden_size=2, n_heads=2)


def test_adapter_branch_does_not_truncate_to_empty_on_newline_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _build_adapter()

    monkeypatch.setattr(
        base._tokenizer,
        "decode",
        lambda token_ids, skip_special_tokens=True: "\nAlice\n",
    )

    out = base.generate(
        "Hello",
        max_new_tokens=2,
        stop="\n",
        adapter=adapter,
        mem_tokens=[2],
        prompt_style="chat",
    )

    assert out["text"].strip() != ""
    assert out["generated_tokens"] >= 1


def test_plain_vs_adapter_stop_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()

    def fake_decode(token_ids, skip_special_tokens=True):  # type: ignore[unused-argument]
        if len(token_ids) > 3:
            return "\nBob\n"
        return "Bob\n"

    monkeypatch.setattr(base._tokenizer, "decode", fake_decode)

    plain = base.generate(
        "Hello",
        max_new_tokens=2,
        stop="\n",
        prompt_style="chat",
    )
    with_adapter = base.generate(
        "Hello",
        max_new_tokens=2,
        stop="\n",
        adapter=adapter,
        mem_tokens=[2],
        prompt_style="chat",
    )

    assert plain["text"].strip() != ""
    assert with_adapter["text"].strip() != ""
