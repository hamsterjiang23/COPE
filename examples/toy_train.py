"""Deterministic CPU smoke run for the COPE training contract."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cope.adapters import StudentAdapter, TeacherAdapter
from cope.config import ProtocolConfig
from cope.objectives import BranchMode, BranchPlanner
from cope.projectors import OrderedPrefixProjector
from cope.readers import TruncatedLinearReader
from cope.reducers import MaskedMeanReducer
from cope.system import CopeSystem
from cope.training import train_step


class ToyTeacher(TeacherAdapter):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding.requires_grad_(False)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        return self.embedding(inputs["input_ids"])


class ToyStudent(StudentAdapter):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def compute_loss(self, prefix_embeddings: Tensor | None, batch: Mapping[str, Tensor]) -> Tensor:
        if prefix_embeddings is None:
            prediction = torch.zeros_like(batch["target"])
        else:
            prediction = prefix_embeddings.mean(dim=1)
        return F.mse_loss(prediction, batch["target"])


def main() -> None:
    torch.manual_seed(7)
    config = ProtocolConfig(teacher_hidden_size=12, max_width=8, widths=(2, 4, 8))
    students = {"small": ToyStudent(6), "medium": ToyStudent(10)}
    system = CopeSystem(
        config,
        ToyTeacher(vocab_size=32, hidden_size=12),
        MaskedMeanReducer(),
        students,
        {
            student_id: TruncatedLinearReader(8, student.hidden_size)
            for student_id, student in students.items()
        },
        shared_projector=OrderedPrefixProjector(12, 8),
    )
    branches = BranchPlanner(config).build(tuple(students), BranchMode.ALL_WIDTHS)
    teacher_inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    student_batches = {
        "small": {"target": torch.randn(2, 6)},
        "medium": {"target": torch.randn(2, 10)},
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in system.parameters() if parameter.requires_grad], lr=1e-3
    )
    metrics = train_step(system, optimizer, teacher_inputs, student_batches, branches, 1.0)
    print(f"loss={metrics.loss:.6f} branches={len(metrics.branch_losses)}")


if __name__ == "__main__":
    main()
