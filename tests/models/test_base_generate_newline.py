"""Regression tests for newline stop handling in ``generate``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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


def test_adapter_branch_strips_prompt_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    prompt = "Hello"

    prompt_ids = base._tokenizer(prompt, return_tensors="pt")["input_ids"].to(base._model.device)
    prompt_prefix = prompt_ids.clone()
    new_token_ids = base._tokenizer("Answer", add_special_tokens=False)["input_ids"]
    new_tokens = torch.tensor(
        [new_token_ids], dtype=prompt_prefix.dtype, device=prompt_prefix.device
    )

    def fake_generate(*args, **kwargs):  # type: ignore[unused-argument]
        return torch.cat([prompt_prefix, new_tokens], dim=1)

    monkeypatch.setattr(base._model, "generate", fake_generate)

    out = base.generate(
        prompt,
        max_new_tokens=len(new_token_ids),
        adapter=adapter,
        mem_tokens=[2],
    )

    expected = base._tokenizer.decode(new_token_ids, skip_special_tokens=True)
    assert out["text"] == expected
    assert out["generated_tokens"] == len(new_token_ids)
