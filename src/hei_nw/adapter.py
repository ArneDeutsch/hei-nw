"""Episodic adapter module (read-only cross-attention)."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class EpisodicAdapter(nn.Module):
    """Cross-attention adapter over episodic memory tokens.

    The adapter is **read-only**: it does not modify any external state and
    simply attends over provided memory tokens ``M_t`` to produce an updated
    hidden representation. When no memory is supplied the input sequence is
    returned *unchanged*.

    Parameters
    ----------
    hidden_size:
        Dimensionality of the model's hidden representations.
    n_heads:
        Number of attention heads.
    dropout:
        Dropout probability applied to the attention output.
    """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_t: Tensor,
        M_t: Tensor | None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply cross-attention over episodic memory tokens.

        Parameters
        ----------
        H_t:
            Hidden states from the base model with shape ``[batch, seq, hidden]``.
        M_t:
            Memory tokens with shape ``[batch, mem, hidden]`` or ``None``.
        attn_mask:
            Optional attention mask broadcastable to the attention weights.

        Returns
        -------
        torch.Tensor
            Updated hidden states. When ``M_t`` is ``None`` or has zero length,
            this is exactly ``H_t`` (same object, no dropout or layer norm
            applied).
        """

        if M_t is None or M_t.shape[1] == 0:
            return H_t

        query = self.ln(H_t)
        memory = self.ln(M_t)
        attn_out, _ = self.attn(query, memory, memory, attn_mask=attn_mask)
        return cast(Tensor, H_t + self.dropout(attn_out))
