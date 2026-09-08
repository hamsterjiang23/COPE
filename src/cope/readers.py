"""Student-specific readers for ordered latent prefixes."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _validate_latent(latent: Tensor, width: int, max_width: int) -> None:
    if latent.ndim != 3 or latent.shape[-1] != max_width:
        raise ValueError(f"latent must have shape [batch, slots, {max_width}]")
    if not 0 < width <= max_width:
        raise ValueError(f"width must be in [1, {max_width}]")


class StudentReader(nn.Module, ABC):
    """Base type for modules mapping latent prefixes into student embeddings."""

    max_width: int
    output_size: int

    @abstractmethod
    def forward(self, maximum_latent: Tensor, width: int) -> Tensor:
        """Map one leading prefix of the maximum latent to student embeddings."""


class TruncatedLinearReader(StudentReader):
    """MRL-E-style reader sharing one maximum weight matrix across widths."""

    def __init__(self, max_width: int, output_size: int) -> None:
        super().__init__()
        if max_width <= 0 or output_size <= 0:
            raise ValueError("max_width and output_size must be positive")
        self.max_width = max_width
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, max_width))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.max_width)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, maximum_latent: Tensor, width: int) -> Tensor:
        _validate_latent(maximum_latent, width, self.max_width)
        return F.linear(maximum_latent[..., :width], self.weight[:, :width], self.bias)


class WidthSpecificReader(StudentReader):
    """MRL-style baseline with an independent linear reader for every width."""

    def __init__(self, widths: tuple[int, ...], output_size: int) -> None:
        super().__init__()
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("widths must contain positive values")
        if len(set(widths)) != len(widths):
            raise ValueError("widths must be unique")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.widths = tuple(sorted(widths))
        self.max_width = self.widths[-1]
        self.output_size = output_size
        self.readers = nn.ModuleDict(
            {str(width): nn.Linear(width, output_size) for width in self.widths}
        )

    def forward(self, maximum_latent: Tensor, width: int) -> Tensor:
        _validate_latent(maximum_latent, width, self.max_width)
        if width not in self.widths:
            raise ValueError(f"width {width} is not configured; expected one of {self.widths}")
        return self.readers[str(width)](maximum_latent[..., :width])
