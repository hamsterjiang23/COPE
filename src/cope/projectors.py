"""Maximum-width latent projectors."""

from __future__ import annotations

from torch import Tensor, nn


class OrderedPrefixProjector(nn.Module):
    """Produces one maximum latent whose exact leading slices define all widths."""

    def __init__(
        self,
        input_size: int,
        max_width: int,
        hidden_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size <= 0 or max_width <= 0:
            raise ValueError("input_size and max_width must be positive")
        hidden_size = hidden_size or max(input_size, max_width)
        self.input_size = input_size
        self.max_width = max_width
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max_width),
        )

    def forward(self, reduced_states: Tensor) -> Tensor:
        if reduced_states.ndim != 3 or reduced_states.shape[-1] != self.input_size:
            raise ValueError(f"reduced_states must have shape [batch, slots, {self.input_size}]")
        return self.network(reduced_states)

    def prefix(self, maximum_latent: Tensor, width: int) -> Tensor:
        if maximum_latent.ndim != 3 or maximum_latent.shape[-1] != self.max_width:
            raise ValueError(f"maximum_latent must have shape [batch, slots, {self.max_width}]")
        if not 0 < width <= self.max_width:
            raise ValueError(f"width must be in [1, {self.max_width}]")
        return maximum_latent[..., :width]
