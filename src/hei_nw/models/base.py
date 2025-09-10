"""Utilities to load the base language model and run generation."""

# mypy: ignore-errors

from __future__ import annotations

from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextGenerationPipeline,
    pipeline,
)

_tokenizer: PreTrainedTokenizerBase | None = None
_model: PreTrainedModel | None = None
_pipe: TextGenerationPipeline | None = None


def load_base(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    dtype: str | torch.dtype = "auto",
    quant_4bit: bool = True,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline]:
    """Load the base model, tokenizer, and text-generation pipeline.

    Parameters
    ----------
    model_id:
        Identifier for the Hugging Face model to load.
    dtype:
        Torch dtype or string alias; ``"auto"`` lets transformers choose.
    quant_4bit:
        Whether to attempt 4-bit quantization via bitsandbytes.

    Returns
    -------
    tuple
        ``(tokenizer, model, pipeline)`` objects ready for generation.
    """
    global _tokenizer, _model, _pipe
    if _tokenizer is not None and _model is not None and _pipe is not None:
        return _tokenizer, _model, _pipe

    torch_dtype: torch.dtype | str = dtype
    if isinstance(dtype, str) and dtype != "auto":
        torch_dtype = getattr(torch, dtype)

    model_kwargs: dict[str, object] = {"device_map": "auto"}
    if dtype == "auto":
        model_kwargs["torch_dtype"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype
    if quant_4bit:
        model_kwargs["load_in_4bit"] = True

    _tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(model_id),  # type: ignore[no-untyped-call]
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call]
            model_id, **model_kwargs
        ),
    )
    if _model.config.pad_token_id is None:
        _model.config.pad_token_id = _tokenizer.pad_token_id

    _pipe = pipeline(  # type: ignore[no-untyped-call]
        "text-generation", model=_model, tokenizer=_tokenizer
    )
    return _tokenizer, _model, _pipe


def generate(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
    stop: str | None = None,
) -> dict[str, int | str]:
    """Generate text using the loaded base model.

    Parameters
    ----------
    prompt:
        Input prompt string.
    max_new_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature.
    top_p:
        Nucleus sampling probability.
    do_sample:
        Whether to enable sampling; otherwise greedy.
    stop:
        Optional substring at which generation should stop.

    Returns
    -------
    dict
        ``{"text", "prompt_tokens", "generated_tokens"}``
    """
    if _model is None or _tokenizer is None:
        load_base()
    if _model is None or _tokenizer is None:  # pragma: no cover - defensive
        raise RuntimeError("Base model is not loaded")

    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    output_ids = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    generated_ids = output_ids[0][prompt_len:]
    text = _tokenizer.decode(generated_ids, skip_special_tokens=True)

    if stop:
        stop_idx = text.find(stop)
        if stop_idx != -1:
            text = text[:stop_idx]
            generated_ids = _tokenizer(text, add_special_tokens=False)["input_ids"]

    result = {
        "text": text,
        "prompt_tokens": int(prompt_len),
        "generated_tokens": int(len(generated_ids)),
    }
    return result
