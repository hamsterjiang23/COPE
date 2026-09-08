"""Causal controls for testing whether students use input-specific latents."""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor, nn


class ControlMode(str, Enum):
    CORRECT = "correct"
    NO_LATENT = "no_latent"
    ZERO = "zero"
    CONSTANT = "constant"
    MISMATCHED = "mismatched"


class LatentController(nn.Module):
    """Applies matched controls without changing the latent tensor contract."""

    def __init__(self, num_slots: int, max_width: int) -> None:
        super().__init__()
        if num_slots <= 0 or max_width <= 0:
            raise ValueError("num_slots and max_width must be positive")
        self.constant = nn.Parameter(torch.zeros(1, num_slots, max_width))

    def forward(self, latent: Tensor, mode: ControlMode) -> Tensor | None:
        if latent.ndim != 3 or latent.shape[1:] != self.constant.shape[1:]:
            raise ValueError(
                "latent shape must match controller slots and maximum width: "
                f"expected [batch, {self.constant.shape[1]}, {self.constant.shape[2]}]"
            )
        if mode is ControlMode.NO_LATENT:
            return None
        if mode is ControlMode.CORRECT:
            return latent
        if mode is ControlMode.ZERO:
            return torch.zeros_like(latent)
        if mode is ControlMode.CONSTANT:
            return self.constant.expand(latent.shape[0], -1, -1)
        if mode is ControlMode.MISMATCHED:
            if latent.shape[0] < 2:
                raise ValueError("mismatched control requires batch size >= 2")
            return latent.roll(shifts=1, dims=0)
        raise ValueError(f"unsupported control mode: {mode}")
