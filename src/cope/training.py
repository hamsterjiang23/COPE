"""Small training primitives shared by exact and sampled objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from cope.objectives import BranchSpec
from cope.system import CopeSystem


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    grad_norm: float | None
    branch_losses: dict[str, float]


def train_step(
    system: CopeSystem,
    optimizer: torch.optim.Optimizer,
    teacher_inputs: Mapping[str, Tensor],
    student_batches: Mapping[str, Mapping[str, Tensor]],
    branches: tuple[BranchSpec, ...],
    max_grad_norm: float | None = None,
) -> StepMetrics:
    """Run one optimizer step over a planned set of student-width branches."""

    system.train()
    optimizer.zero_grad(set_to_none=True)
    output = system(teacher_inputs, student_batches, branches)
    if not output.loss.requires_grad:
        raise RuntimeError(
            "loss has no trainable path; unfreeze student adaptation parameters for a "
            "standalone no-latent baseline"
        )
    output.loss.backward()
    grad_norm: float | None = None
    if max_grad_norm is not None:
        parameters = [parameter for parameter in system.parameters() if parameter.requires_grad]
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        grad_norm = float(grad_norm_tensor.item())
    optimizer.step()
    return StepMetrics(
        loss=float(output.loss.detach().item()),
        grad_norm=grad_norm,
        branch_losses={
            key: float(loss.detach().item()) for key, loss in output.branch_losses.items()
        },
    )


@torch.no_grad()
def evaluate_step(
    system: CopeSystem,
    teacher_inputs: Mapping[str, Tensor],
    student_batches: Mapping[str, Mapping[str, Tensor]],
    branches: tuple[BranchSpec, ...],
) -> StepMetrics:
    system.eval()
    output = system(teacher_inputs, student_batches, branches)
    return StepMetrics(
        loss=float(output.loss.item()),
        grad_norm=None,
        branch_losses={key: float(loss.item()) for key, loss in output.branch_losses.items()},
    )
