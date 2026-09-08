"""Validated protocol configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolConfig:
    """Shape contract for one ordered latent protocol."""

    teacher_hidden_size: int
    max_width: int
    widths: tuple[int, ...]
    num_slots: int = 1

    def __post_init__(self) -> None:
        if self.teacher_hidden_size <= 0:
            raise ValueError("teacher_hidden_size must be positive")
        if self.max_width <= 0:
            raise ValueError("max_width must be positive")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if not self.widths:
            raise ValueError("widths must not be empty")
        if any(width <= 0 for width in self.widths):
            raise ValueError("all widths must be positive")
        if tuple(sorted(set(self.widths))) != self.widths:
            raise ValueError("widths must be unique and strictly increasing")
        if self.widths[-1] != self.max_width:
            raise ValueError("widths must end at max_width")

    def validate_width(self, width: int) -> None:
        if width not in self.widths:
            raise ValueError(f"width {width} is not configured; expected one of {self.widths}")
