"""Adapters between COPE tensors and model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn


class TeacherAdapter(nn.Module, ABC):
    """Returns token-level teacher states with shape [batch, sequence, hidden]."""

    @abstractmethod
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Encode one teacher input batch."""


class StudentAdapter(nn.Module, ABC):
    """Consumes optional prefix embeddings and returns a scalar task loss."""

    hidden_size: int

    @abstractmethod
    def compute_loss(self, prefix_embeddings: Tensor | None, batch: Mapping[str, Tensor]) -> Tensor:
        """Compute one student branch loss."""


class HuggingFaceTeacherAdapter(TeacherAdapter):
    """Extracts a hidden-state layer from a Hugging Face-compatible model."""

    def __init__(self, model: nn.Module, hidden_state_index: int = -1, freeze: bool = True) -> None:
        super().__init__()
        self.model = model
        self.hidden_state_index = hidden_state_index
        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> HuggingFaceTeacherAdapter:
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        model_inputs = dict(inputs)
        model_inputs.pop("labels", None)
        model_inputs.update(output_hidden_states=True, use_cache=False, return_dict=True)
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            outputs = self.model(**model_inputs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            result = hidden_states[self.hidden_state_index]
        else:
            result = getattr(outputs, "last_hidden_state", None)
        if result is None:
            raise TypeError("teacher output must expose hidden_states or last_hidden_state")
        if result.ndim != 3:
            raise ValueError("teacher hidden state must have shape [batch, sequence, hidden]")
        return result


class HuggingFaceCausalStudentAdapter(StudentAdapter):
    """Prepends reader outputs through `inputs_embeds` for causal-LM training."""

    def __init__(self, model: nn.Module, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.model = model
        self.freeze_backbone = freeze_backbone
        embedding = self._embedding_layer()
        self.hidden_size = int(embedding.weight.shape[-1])
        if freeze_backbone:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> HuggingFaceCausalStudentAdapter:
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def _embedding_layer(self) -> nn.Module:
        getter = getattr(self.model, "get_input_embeddings", None)
        if getter is None:
            raise TypeError("student model must define get_input_embeddings()")
        embedding = getter()
        if embedding is None or not hasattr(embedding, "weight"):
            raise TypeError("student input embedding must expose a weight tensor")
        return embedding

    def _prepare_inputs(
        self, prefix_embeddings: Tensor | None, batch: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        if "input_ids" not in batch:
            raise KeyError("student batch must contain input_ids")
        input_ids = batch["input_ids"]
        token_embeddings = self._embedding_layer()(input_ids)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        labels = batch.get("labels", input_ids)

        if prefix_embeddings is None:
            return {
                "inputs_embeds": token_embeddings,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": (attention_mask.cumsum(-1) - 1).clamp_min(0),
            }
        if prefix_embeddings.ndim != 3:
            raise ValueError("prefix_embeddings must have shape [batch, slots, hidden]")
        if prefix_embeddings.shape[0] != input_ids.shape[0]:
            raise ValueError("prefix and student batch sizes must match")
        if prefix_embeddings.shape[-1] != self.hidden_size:
            raise ValueError(
                f"prefix hidden size must be {self.hidden_size}, got {prefix_embeddings.shape[-1]}"
            )
        prefix_embeddings = prefix_embeddings.to(
            device=token_embeddings.device, dtype=token_embeddings.dtype
        )

        prefix_length = prefix_embeddings.shape[1]
        prefix_mask = torch.ones(
            input_ids.shape[0],
            prefix_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        ignored_prefix_labels = torch.full(
            (input_ids.shape[0], prefix_length),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return {
            "inputs_embeds": torch.cat([prefix_embeddings, token_embeddings], dim=1),
            "attention_mask": combined_mask,
            "position_ids": (combined_mask.cumsum(-1) - 1).clamp_min(0),
            "labels": torch.cat([ignored_prefix_labels, labels], dim=1),
        }

    def compute_loss(self, prefix_embeddings: Tensor | None, batch: Mapping[str, Tensor]) -> Tensor:
        model_inputs: dict[str, Any] = self._prepare_inputs(prefix_embeddings, batch)
        model_inputs.update(use_cache=False, return_dict=True)
        outputs = self.model(**model_inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None or loss.ndim != 0:
            raise TypeError("student model output must expose a scalar loss")
        return loss

    @torch.no_grad()
    def generate(
        self,
        prefix_embeddings: Tensor | None,
        batch: Mapping[str, Tensor],
        **generation_kwargs: Any,
    ) -> Any:
        model_inputs = self._prepare_inputs(prefix_embeddings, batch)
        model_inputs.pop("labels")
        # GenerationMixin must extend positions as new tokens are generated.
        model_inputs.pop("position_ids")
        generate = getattr(self.model, "generate", None)
        if generate is None:
            raise TypeError("student model must define generate()")
        return generate(**model_inputs, **generation_kwargs)
