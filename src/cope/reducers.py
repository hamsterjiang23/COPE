"""Teacher sequence reducers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _validate_inputs(hidden_states: Tensor, attention_mask: Tensor | None) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if attention_mask is not None and attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, sequence]")


class SequenceReducer(ABC, nn.Module):
    """Maps variable-length teacher states to a fixed number of latent slots."""

    @abstractmethod
    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return states with shape [batch, slots, hidden]."""


class MaskedMeanReducer(SequenceReducer):
    """Mean-pools valid tokens into contiguous adaptive slots."""

    def __init__(self, num_slots: int = 1) -> None:
        super().__init__()
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _validate_inputs(hidden_states, attention_mask)
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool
            )
        else:
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)

        pooled: list[Tensor] = []
        for states, mask in zip(hidden_states, attention_mask, strict=True):
            valid_states = states[mask]
            if valid_states.shape[0] == 0:
                raise ValueError("every sample must contain at least one unmasked token")
            slot_states = F.adaptive_avg_pool1d(
                valid_states.transpose(0, 1).unsqueeze(0), self.num_slots
            )
            pooled.append(slot_states.squeeze(0).transpose(0, 1))
        return torch.stack(pooled)


class LearnedQueryReducer(SequenceReducer):
    """Cross-attends learned slot queries to the teacher sequence."""

    def __init__(self, hidden_size: int, num_slots: int, num_heads: int = 1) -> None:
        super().__init__()
        if hidden_size <= 0 or num_slots <= 0 or num_heads <= 0:
            raise ValueError("hidden_size, num_slots, and num_heads must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.queries = nn.Parameter(torch.empty(num_slots, hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        nn.init.normal_(self.queries, std=hidden_size**-0.5)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _validate_inputs(hidden_states, attention_mask)
        batch_size = hidden_states.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            if key_padding_mask.all(dim=1).any():
                raise ValueError("every sample must contain at least one unmasked token")
        reduced, _ = self.attention(
            queries,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return reduced
