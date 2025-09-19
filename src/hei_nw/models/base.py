"""Utilities to load the base language model and run generation."""

# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextGenerationPipeline,
    pipeline,
)
from transformers.utils import logging as hf_logging

from hei_nw.adapter import EpisodicAdapter

hf_logging.set_verbosity_error()

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

    resolved_dtype: torch.dtype | str = dtype
    if isinstance(dtype, str) and dtype != "auto":
        resolved_dtype = getattr(torch, dtype)

    quant_config: BitsAndBytesConfig | None = None
    if quant_4bit:
        quant_config = BitsAndBytesConfig(load_in_4bit=True)

    model_kwargs: dict[str, object] = {"device_map": "auto"}
    if dtype == "auto":
        model_kwargs["dtype"] = "auto"
    else:
        model_kwargs["dtype"] = resolved_dtype
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

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

    gen_conf: GenerationConfig = _model.generation_config
    gen_conf.top_k = 0
    gen_conf.top_p = 1.0
    gen_conf.temperature = 1.0
    gen_conf.do_sample = False

    _pipe = pipeline(  # type: ignore[no-untyped-call]
        "text-generation", model=_model, tokenizer=_tokenizer
    )
    return _tokenizer, _model, _pipe


PromptData = str | Sequence[dict[str, str]]


def _truncate_at_stop(text: str, stop: str | None) -> tuple[str, bool]:
    """Return ``text`` truncated at the first occurrence of ``stop``.

    Parameters
    ----------
    text:
        Generated text to truncate.
    stop:
        Optional stop substring. When ``None`` or empty the input text is
        returned unchanged.

    Returns
    -------
    tuple[str, bool]
        ``(possibly_truncated_text, did_truncate)``
    """

    if not stop:
        return text, False
    index = text.find(stop)
    if index == -1:
        return text, False
    return text[:index], True


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt_or_messages: PromptData,
    prompt_style: str,
    template_policy: str = "auto",
) -> str:
    """Render *prompt_or_messages* according to *prompt_style*."""

    if prompt_style == "chat":
        normalized_policy = template_policy.lower()
        if normalized_policy not in {"auto", "plain"}:
            msg = f"Unsupported template policy: {template_policy}"
            raise ValueError(msg)
        if isinstance(prompt_or_messages, str):
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = [
                {
                    "role": str(msg.get("role", "user")),
                    "content": str(msg.get("content", "")),
                }
                for msg in prompt_or_messages
            ]
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if normalized_policy == "auto" and callable(apply_template):
            rendered: str | list[int] | None
            try:
                rendered = apply_template(  # type: ignore[call-arg]
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except (TypeError, ValueError):
                try:
                    rendered = apply_template(messages, add_generation_prompt=True)
                except (TypeError, ValueError):
                    rendered = None
            if rendered is not None:
                if isinstance(rendered, list):
                    return tokenizer.decode(rendered, skip_special_tokens=True)
                return cast(str, rendered)
        formatted = [
            f"{msg['role'].upper()}: {msg['content']}".strip()
            for msg in messages
            if msg.get("content")
        ]
        formatted.append("ASSISTANT:")
        return "\n\n".join(formatted)

    if not isinstance(prompt_or_messages, str):
        raise TypeError("Plain prompt style expects a string input.")
    return prompt_or_messages


def generate(
    prompt: PromptData,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
    stop: str | None = None,
    *,
    stop_mode: str = "substring",
    mem_tokens: list[int] | None = None,
    adapter: EpisodicAdapter | None = None,
    prompt_style: str = "plain",
    template_policy: str = "auto",
    **kwargs: object,
) -> dict[str, int | str]:
    """Generate text using the loaded base model.

    Parameters
    ----------
    prompt:
        Input prompt string or chat message sequence.
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
    stop_mode:
        Strategy used to apply ``stop``. ``"substring"`` retains the previous
        behaviour; ``"none"`` disables substring truncation.
    mem_tokens:
        Optional list of memory token IDs. When used with ``adapter`` they are
        converted to embeddings and cross-attended during generation.
    adapter:
        Optional ``EpisodicAdapter`` instance applied when ``mem_tokens`` are
        supplied.
    prompt_style:
        Rendering style to apply to ``prompt`` (``"plain"`` or ``"chat"``).
    template_policy:
        Policy controlling chat template application (``"auto"`` or ``"plain"``).

    Returns
    -------
    dict
        ``{"text", "prompt_tokens", "generated_tokens"}``
    """
    if _model is None or _tokenizer is None:
        load_base()
    if _model is None or _tokenizer is None:  # pragma: no cover - defensive
        raise RuntimeError("Base model is not loaded")

    prompt_text = build_prompt(
        _tokenizer, prompt, prompt_style, template_policy=template_policy
    )
    inputs = _tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    gen_kwargs: dict[str, object | None] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    gen_kwargs.update(kwargs)

    if adapter is not None and mem_tokens:
        mem_ids = torch.tensor([mem_tokens], dtype=input_ids.dtype, device=_model.device)
        mem_embeds = _model.get_input_embeddings()(mem_ids)
        prompt_embeds = _model.get_input_embeddings()(input_ids)
        adapted = adapter(prompt_embeds, mem_embeds)
        if adapted.shape[-2] > 0:
            # Keep the conversational context untouched and only let the adapter
            # steer the final assistant token that seeds generation.
            adapted = adapted.clone()
            adapted[:, :-1, :] = prompt_embeds[:, :-1, :]
        gen_input = dict(inputs)
        gen_input["inputs_embeds"] = adapted
    else:
        gen_input = inputs

    output_ids = _model.generate(**gen_input, pad_token_id=_tokenizer.pad_token_id, **gen_kwargs)
    generated_ids = output_ids[0]
    prefix_stripped = False

    if adapter is not None and mem_tokens:
        if generated_ids.shape[-1] > prompt_len:
            generated_ids = generated_ids[prompt_len:]
            prefix_stripped = True
    else:
        generated_ids = generated_ids[prompt_len:]
        prefix_stripped = True
    text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
    retokenize = False

    if adapter is not None and mem_tokens:
        # Chat templates often leave a leading newline at the assistant turn boundary; trim it
        # so stop handling does not erase the entire output.
        stripped_text = text.lstrip()
        if stripped_text != text:
            text = stripped_text
            retokenize = True

    normalized_stop_mode = stop_mode.lower()
    if normalized_stop_mode not in {"substring", "none"}:
        msg = f"Unsupported stop mode: {stop_mode}"
        raise ValueError(msg)

    if normalized_stop_mode == "substring":
        text, did_truncate = _truncate_at_stop(text, stop)
        if did_truncate:
            retokenize = True
    else:
        did_truncate = False

    if retokenize:
        generated_ids = _tokenizer(text, add_special_tokens=False)["input_ids"]

    result = {
        "text": text,
        "prompt_tokens": int(prompt_len),
        "generated_tokens": int(len(generated_ids)),
        "prefix_stripped": prefix_stripped,
    }
    return result


def build_default_adapter(
    model: PreTrainedModel, *, scale: float = 0.2
) -> EpisodicAdapter:
    """Construct a default ``EpisodicAdapter`` matching ``model`` geometry."""

    hidden_size = getattr(model.config, "hidden_size", None)
    n_heads = getattr(model.config, "num_attention_heads", None)
    if hidden_size is None or n_heads is None:
        raise ValueError("Model config lacks hidden_size or num_attention_heads")

    adapter = EpisodicAdapter(hidden_size=hidden_size, n_heads=n_heads, scale=scale)
    model_device = getattr(model, "device", None)
    model_dtype = getattr(model, "dtype", None)

    if model_device is None or model_dtype is None:
        try:
            param = next(model.parameters())
            if model_device is None:
                model_device = param.device
            if model_dtype is None:
                model_dtype = param.dtype
        except (StopIteration, AttributeError):  # pragma: no cover - defensive
            pass

    to_kwargs: dict[str, object] = {}
    if model_device is not None:
        to_kwargs["device"] = model_device
    if model_dtype is not None:
        to_kwargs["dtype"] = model_dtype

    return adapter.to(**to_kwargs) if to_kwargs else adapter
