from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cope.adapters import StudentAdapter, TeacherAdapter


class CountingTeacher(TeacherAdapter):
    def __init__(self, hidden_size: int = 6, vocab_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.calls = 0

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        self.calls += 1
        return self.embedding(inputs["input_ids"])


class RecordingStudent(StudentAdapter):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.prefixes: list[Tensor | None] = []

    def compute_loss(self, prefix_embeddings: Tensor | None, batch: Mapping[str, Tensor]) -> Tensor:
        self.prefixes.append(prefix_embeddings)
        if prefix_embeddings is None:
            prediction = torch.zeros_like(batch["target"])
        else:
            prediction = prefix_embeddings.mean(dim=1)
        return F.mse_loss(prediction, batch["target"])


class FakeTeacherModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.calls = 0

    def forward(self, input_ids: Tensor, **_: object) -> SimpleNamespace:
        self.calls += 1
        states = self.embedding(input_ids)
        return SimpleNamespace(hidden_states=(states * 0.5, states))


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.last_inputs: dict[str, Tensor] = {}

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        **_: object,
    ) -> SimpleNamespace:
        self.last_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        logits = self.projection(inputs_embeds)
        loss = logits.square().mean()
        return SimpleNamespace(loss=loss)
