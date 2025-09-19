"""Tests for prompt prefix stripping when using ``inputs_embeds``."""

from __future__ import annotations

from pathlib import Path
import torch
import pytest

import hei_nw.models.base as base
from hei_nw.adapter import EpisodicAdapter

TINY_MODEL = Path(__file__).resolve().parents[2] / "models" / "tiny-gpt2"


def setup_module() -> None:
    base.load_base(model_id=str(TINY_MODEL), quant_4bit=False)


def test_inputs_embeds_path_strips_prompt_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = EpisodicAdapter(hidden_size=2, n_heads=2)
    prompt = "Who are you?"

    tokenizer = base._tokenizer
    model = base._model
    assert tokenizer is not None and model is not None

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    new_token_ids = tokenizer("Answer", add_special_tokens=False)["input_ids"]
    new_tokens = torch.tensor([new_token_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)

    def fake_generate(*args, **kwargs):  # type: ignore[unused-argument]
        assert "inputs_embeds" in kwargs, "expected inputs_embeds when adapter is used"
        input_ids = kwargs["input_ids"]
        return torch.cat([input_ids, new_tokens], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)

    out = base.generate(
        prompt,
        max_new_tokens=len(new_token_ids),
        adapter=adapter,
        mem_tokens=[1, 2],
        prompt_style="plain",
    )

    expected_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    assert out["text"] == expected_text
    assert out["generated_tokens"] == len(new_token_ids)
    assert out["prefix_stripped"] is True
