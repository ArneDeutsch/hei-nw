"""Deterministic dummy model used for fast tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

DUMMY_MODEL_ID = "hei-nw/dummy-model"


class DummyTokenizer:
    """Minimal tokenizer with a mutable vocabulary."""

    pad_token = "<pad>"  # noqa: S105 - placeholder token for deterministic tests
    eos_token = "<eos>"  # noqa: S105 - placeholder token for deterministic tests

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab: dict[str, int] = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
        }
        self._inv_vocab: dict[int, str] = {
            self.pad_token_id: self.pad_token,
            self.eos_token_id: self.eos_token,
        }

    def _tokenise(self, text: str) -> list[str]:
        tokens = text.strip().split()
        return tokens if tokens else []

    def _ensure_token(self, token: str) -> int:
        if token not in self._vocab:
            idx = len(self._vocab)
            self._vocab[token] = idx
            self._inv_vocab[idx] = token
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = self._tokenise(text)
        ids = [self._ensure_token(tok) for tok in tokens]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int | Tensor], skip_special_tokens: bool = True) -> str:
        indices: list[int] = []
        for idx in ids:
            if isinstance(idx, Tensor):
                index = int(idx.item())
            else:
                index = int(idx)
            if skip_special_tokens and index in {self.pad_token_id, self.eos_token_id}:
                continue
            indices.append(index)
        tokens = [self._inv_vocab.get(index, "<unk>") for index in indices]
        return " ".join(tokens).strip()

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str = False,
        **_: object,
    ) -> dict[str, list[int] | list[list[int]] | Tensor]:
        if isinstance(text, Sequence) and not isinstance(text, str):
            batch = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
            if padding:
                pad_length = max(len(row) for row in batch) if batch else 0
                padded: list[list[int]] = []
                for row in batch:
                    pad_row = list(row)
                    pad_row.extend([self.pad_token_id] * (pad_length - len(pad_row)))
                    padded.append(pad_row)
                batch = padded
            if return_tensors == "pt":
                return {"input_ids": torch.tensor(batch, dtype=torch.long)}
            return {"input_ids": batch}

        ids = self.encode(str(text), add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower() or "user"
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        if add_generation_prompt:
            lines.append("assistant:")
        rendered = "\n".join(lines) if lines else "assistant:"
        if tokenize:
            return self.encode(rendered)
        return rendered


class _DynamicEmbedding(nn.Module):
    """Embedding layer that grows with the tokenizer vocabulary."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:  # pragma: no cover - torch contract
        self._ensure_capacity(int(input_ids.max().item()) + 1)
        return cast(Tensor, self.embedding(input_ids))

    def _ensure_capacity(self, size: int) -> None:
        if size <= self.embedding.num_embeddings:
            return
        old_weight = self.embedding.weight.data.clone()
        new_embedding = nn.Embedding(size, self.embedding.embedding_dim)
        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_embedding.weight[: old_weight.shape[0]].copy_(old_weight)
        self.embedding = new_embedding


@dataclass
class DummyConfig:
    """Configuration stub mimicking HF models."""

    num_hidden_layers: int = 2
    hidden_size: int = 2
    num_attention_heads: int = 2
    dtype: torch.dtype = torch.float32
    pad_token_id: int = 0


class DummyModel(nn.Module):
    """Minimal causal language model used by tests."""

    def __init__(self, tokenizer: DummyTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = DummyConfig(pad_token_id=tokenizer.pad_token_id)
        self._embeddings = _DynamicEmbedding(embed_dim=self.config.hidden_size)

    def get_input_embeddings(self) -> nn.Module:
        return self._embeddings

    @property
    def device(self) -> torch.device:
        return self._embeddings.embedding.weight.device

    def generate(  # noqa: D401 - matches transformers signature subset
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 1,
        pad_token_id: int | None = None,
        inputs_embeds: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        if pad_token_id is not None and pad_token_id != self.tokenizer.pad_token_id:
            self._embeddings(torch.tensor([[pad_token_id]], device=input_ids.device))
        prompt = input_ids.to(self.device)

        if inputs_embeds is not None:
            reference = self.get_input_embeddings()(prompt.to(self.device))
            if torch.allclose(inputs_embeds.to(self.device), reference, atol=1e-5, rtol=1e-5):
                response = "stairs"
            else:
                response = "factors"
        else:
            response = "stairs"
        base_ids = self.tokenizer.encode(response, add_special_tokens=False)
        if not base_ids:
            base_ids = [self.tokenizer.eos_token_id]
        pattern = torch.tensor(base_ids, device=self.device, dtype=prompt.dtype)
        if max_new_tokens <= 0:
            generated_core = torch.empty(0, device=self.device, dtype=prompt.dtype)
        else:
            repeats = (max_new_tokens + len(pattern) - 1) // len(pattern)
            generated_core = pattern.repeat(repeats)[:max_new_tokens]

        outputs: list[Tensor] = []
        for row in prompt:
            outputs.append(torch.cat([row, generated_core], dim=-1))
        return torch.stack(outputs)


class DummyPipeline:
    """Simplified text-generation pipeline."""

    def __init__(self, tokenizer: DummyTokenizer, model: DummyModel) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, prompt: str, max_new_tokens: int = 16, **_: object) -> list[dict[str, str]]:
        encoded = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        inputs = cast(Tensor, encoded).to(self.model.device)
        output = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs.shape[-1]
        gen_ids = output[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids)
        return [{"generated_text": text}]


_components: tuple[DummyTokenizer, DummyModel, DummyPipeline] | None = None


def create_dummy_components() -> tuple[DummyTokenizer, DummyModel, DummyPipeline]:
    """Return freshly constructed dummy components."""

    tokenizer = DummyTokenizer()
    model = DummyModel(tokenizer)
    model.eval()
    pipeline = DummyPipeline(tokenizer, model)
    return tokenizer, model, pipeline


def load_dummy_components() -> tuple[DummyTokenizer, DummyModel, DummyPipeline]:
    """Load and cache dummy components for reuse."""

    global _components
    if _components is None:
        _components = create_dummy_components()
    return _components


def is_dummy_model_id(model_id: str) -> bool:
    """Return ``True`` when *model_id* references the dummy model."""

    return model_id.strip().lower() in {DUMMY_MODEL_ID, "dummy", "dummy-local"}
