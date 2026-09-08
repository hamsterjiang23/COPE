"""Student-width branch planning and MRL-style loss aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import torch
from torch import Tensor

from cope.config import ProtocolConfig
from cope.controls import ControlMode


class BranchMode(str, Enum):
    FULL_ONLY = "full_only"
    ALL_WIDTHS = "all_widths"
    SAMPLE_WIDTH = "sample_width"


@dataclass(frozen=True)
class BranchSpec:
    student_id: str
    width: int
    weight: float = 1.0
    control: ControlMode = ControlMode.CORRECT

    @property
    def key(self) -> str:
        return f"{self.student_id}/d={self.width}/{self.control.value}"


class BranchPlanner:
    """Builds exact or unbiased sampled approximations to the width objective."""

    def __init__(self, config: ProtocolConfig) -> None:
        self.config = config

    def build(
        self,
        student_ids: tuple[str, ...],
        mode: BranchMode,
        weights: dict[tuple[str, int], float] | None = None,
        control: ControlMode = ControlMode.CORRECT,
        generator: torch.Generator | None = None,
    ) -> tuple[BranchSpec, ...]:
        if not student_ids or len(set(student_ids)) != len(student_ids):
            raise ValueError("student_ids must be non-empty and unique")
        weights = weights or {}
        branches: list[BranchSpec] = []
        for student_id in student_ids:
            if mode is BranchMode.FULL_ONLY:
                selected = (self.config.max_width,)
                probability_correction = 1.0
            elif mode is BranchMode.ALL_WIDTHS:
                selected = self.config.widths
                probability_correction = 1.0
            elif mode is BranchMode.SAMPLE_WIDTH:
                sampled_index = int(
                    torch.randint(len(self.config.widths), (), generator=generator).item()
                )
                selected = (self.config.widths[sampled_index],)
                probability_correction = float(len(self.config.widths))
            else:
                raise ValueError(f"unsupported branch mode: {mode}")

            for width in selected:
                weight = weights.get((student_id, width), 1.0) * probability_correction
                if not isfinite(weight) or weight < 0:
                    raise ValueError("branch weights must be finite and non-negative")
                branches.append(BranchSpec(student_id, width, weight, control))
        return tuple(branches)


def aggregate_weighted_losses(
    branches: tuple[BranchSpec, ...], branch_losses: dict[str, Tensor]
) -> Tensor:
    """Return the MRL-style weighted sum over every requested branch."""

    if not branches:
        raise ValueError("at least one branch is required")
    missing = [branch.key for branch in branches if branch.key not in branch_losses]
    if missing:
        raise KeyError(f"missing losses for branches: {missing}")
    non_scalars = [key for key, loss in branch_losses.items() if loss.ndim != 0]
    if non_scalars:
        raise ValueError(f"branch losses must be scalar: {non_scalars}")
    return torch.stack([branch_losses[branch.key] * branch.weight for branch in branches]).sum()
