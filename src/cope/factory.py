"""Factories for assembling COPE around Hugging Face-compatible models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from torch import nn

from cope.adapters import HuggingFaceCausalStudentAdapter, HuggingFaceTeacherAdapter
from cope.config import ProtocolConfig
from cope.projectors import OrderedPrefixProjector
from cope.readers import StudentReader, TruncatedLinearReader, WidthSpecificReader
from cope.reducers import LearnedQueryReducer, MaskedMeanReducer, SequenceReducer
from cope.system import CopeSystem


def _build_reducer(
    config: ProtocolConfig, kind: Literal["masked_mean", "learned_query"], num_heads: int
) -> SequenceReducer:
    if kind == "masked_mean":
        return MaskedMeanReducer(config.num_slots)
    if kind == "learned_query":
        return LearnedQueryReducer(
            config.teacher_hidden_size, config.num_slots, num_heads=num_heads
        )
    raise ValueError(f"unknown reducer kind: {kind}")


def _build_reader(
    config: ProtocolConfig,
    output_size: int,
    kind: Literal["truncated_shared", "width_specific"],
) -> StudentReader:
    if kind == "truncated_shared":
        return TruncatedLinearReader(config.max_width, output_size)
    if kind == "width_specific":
        return WidthSpecificReader(config.widths, output_size)
    raise ValueError(f"unknown reader kind: {kind}")


def build_huggingface_system(
    config: ProtocolConfig,
    teacher_model: nn.Module,
    student_models: Mapping[str, nn.Module],
    *,
    projector_sharing: Literal["shared", "private"] = "shared",
    reader_kind: Literal["truncated_shared", "width_specific"] = "truncated_shared",
    reducer_kind: Literal["masked_mean", "learned_query"] = "masked_mean",
    reducer_num_heads: int = 1,
    projector_hidden_size: int | None = None,
    projector_dropout: float = 0.0,
    teacher_hidden_state_index: int = -1,
    freeze_teacher: bool = True,
    freeze_student_backbones: bool = True,
) -> CopeSystem:
    """Construct a validated system without depending on concrete Transformers classes."""

    if not student_models:
        raise ValueError("student_models must not be empty")
    teacher = HuggingFaceTeacherAdapter(
        teacher_model, hidden_state_index=teacher_hidden_state_index, freeze=freeze_teacher
    )
    students = {
        student_id: HuggingFaceCausalStudentAdapter(model, freeze_backbone=freeze_student_backbones)
        for student_id, model in student_models.items()
    }
    readers = {
        student_id: _build_reader(config, student.hidden_size, reader_kind)
        for student_id, student in students.items()
    }
    reducer = _build_reducer(config, reducer_kind, reducer_num_heads)

    projector_kwargs = {
        "input_size": config.teacher_hidden_size,
        "max_width": config.max_width,
        "hidden_size": projector_hidden_size,
        "dropout": projector_dropout,
    }
    if projector_sharing == "shared":
        system = CopeSystem(
            config,
            teacher,
            reducer,
            students,
            readers,
            shared_projector=OrderedPrefixProjector(**projector_kwargs),
        )
    elif projector_sharing == "private":
        system = CopeSystem(
            config,
            teacher,
            reducer,
            students,
            readers,
            private_projectors={
                student_id: OrderedPrefixProjector(**projector_kwargs) for student_id in students
            },
        )
    else:
        raise ValueError(f"unknown projector sharing mode: {projector_sharing}")

    reference_parameter = next(teacher_model.parameters(), None)
    if reference_parameter is not None:
        protocol_dtype = (
            reference_parameter.dtype if reference_parameter.is_floating_point() else torch.float32
        )
        system.reducer.to(device=reference_parameter.device, dtype=protocol_dtype)
        system.readers.to(device=reference_parameter.device, dtype=protocol_dtype)
        system.private_projectors.to(device=reference_parameter.device, dtype=protocol_dtype)
        system.controller.to(device=reference_parameter.device, dtype=protocol_dtype)
        if system.shared_projector is not None:
            system.shared_projector.to(device=reference_parameter.device, dtype=protocol_dtype)
    return system
