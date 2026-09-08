"""One-prefill, multi-student, multi-width COPE system."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor, nn

from cope.adapters import StudentAdapter, TeacherAdapter
from cope.config import ProtocolConfig
from cope.controls import ControlMode, LatentController
from cope.objectives import BranchSpec, aggregate_weighted_losses
from cope.projectors import OrderedPrefixProjector
from cope.readers import StudentReader
from cope.reducers import SequenceReducer


@dataclass
class CopeForwardOutput:
    loss: Tensor
    branch_losses: dict[str, Tensor]
    maximum_latents: dict[str, Tensor]


class CopeSystem(nn.Module):
    """Coordinates a shared teacher representation and student-width branches."""

    def __init__(
        self,
        config: ProtocolConfig,
        teacher: TeacherAdapter,
        reducer: SequenceReducer,
        students: Mapping[str, StudentAdapter],
        readers: Mapping[str, StudentReader],
        *,
        shared_projector: OrderedPrefixProjector | None = None,
        private_projectors: Mapping[str, OrderedPrefixProjector] | None = None,
        controller: LatentController | None = None,
    ) -> None:
        super().__init__()
        if not students:
            raise ValueError("at least one student is required")
        student_keys = set(students)
        if student_keys != set(readers):
            raise ValueError("students and readers must have identical keys")
        if (shared_projector is None) == (private_projectors is None):
            raise ValueError("provide exactly one of shared_projector or private_projectors")
        if private_projectors is not None and student_keys != set(private_projectors):
            raise ValueError("private projectors must have the same keys as students")

        self.config = config
        self.teacher = teacher
        self.reducer = reducer
        self.students = nn.ModuleDict(dict(students))
        self.readers = nn.ModuleDict(dict(readers))
        self.shared_projector = shared_projector
        self.private_projectors = nn.ModuleDict(dict(private_projectors or {}))
        self.controller = controller or LatentController(config.num_slots, config.max_width)
        self._validate_modules()

    @property
    def projector_sharing(self) -> str:
        return "shared" if self.shared_projector is not None else "private"

    def _validate_modules(self) -> None:
        projectors = (
            [self.shared_projector]
            if self.shared_projector is not None
            else list(self.private_projectors.values())
        )
        for projector in projectors:
            if projector is None:
                continue
            if projector.input_size != self.config.teacher_hidden_size:
                raise ValueError("projector input size must match teacher_hidden_size")
            if projector.max_width != self.config.max_width:
                raise ValueError("projector max width must match protocol config")
        for student_id, reader in self.readers.items():
            if reader.max_width != self.config.max_width:
                raise ValueError(f"reader {student_id} max width must match protocol config")
            if reader.output_size != self.students[student_id].hidden_size:
                raise ValueError(f"reader {student_id} output size must match student hidden size")

    def _project(self, reduced_states: Tensor, student_ids: set[str]) -> dict[str, Tensor]:
        if self.shared_projector is not None:
            maximum_latent = self.shared_projector(reduced_states)
            return {student_id: maximum_latent for student_id in student_ids}
        return {
            student_id: self.private_projectors[student_id](reduced_states)
            for student_id in student_ids
        }

    def forward(
        self,
        teacher_inputs: Mapping[str, Tensor],
        student_batches: Mapping[str, Mapping[str, Tensor]],
        branches: tuple[BranchSpec, ...],
    ) -> CopeForwardOutput:
        if not branches:
            raise ValueError("at least one branch is required")
        requested_students = {branch.student_id for branch in branches}
        unknown_students = requested_students - set(self.students)
        if unknown_students:
            raise KeyError(f"unknown students: {sorted(unknown_students)}")
        missing_batches = requested_students - set(student_batches)
        if missing_batches:
            raise KeyError(f"missing student batches: {sorted(missing_batches)}")
        for branch in branches:
            self.config.validate_width(branch.width)
        branch_keys = [branch.key for branch in branches]
        if len(set(branch_keys)) != len(branch_keys):
            raise ValueError("branches must have unique student/width/control keys")

        needs_teacher = any(branch.control is not ControlMode.NO_LATENT for branch in branches)
        maximum_latents: dict[str, Tensor] = {}
        if needs_teacher:
            teacher_states = self.teacher(teacher_inputs)
            if teacher_states.shape[-1] != self.config.teacher_hidden_size:
                raise ValueError(
                    "teacher hidden size does not match protocol config: "
                    f"expected {self.config.teacher_hidden_size}, got {teacher_states.shape[-1]}"
                )
            attention_mask = teacher_inputs.get("attention_mask")
            reduced_states = self.reducer(teacher_states, attention_mask)
            if reduced_states.shape[:2] != (
                teacher_states.shape[0],
                self.config.num_slots,
            ):
                raise ValueError(
                    "reducer output must have shape "
                    f"[batch, {self.config.num_slots}, teacher_hidden_size]"
                )
            maximum_latents = self._project(reduced_states, requested_students)
            if any(
                latent.shape
                != (teacher_states.shape[0], self.config.num_slots, self.config.max_width)
                for latent in maximum_latents.values()
            ):
                raise ValueError("projector output does not match the configured latent shape")

        branch_losses: dict[str, Tensor] = {}
        for branch in branches:
            if branch.control is ControlMode.NO_LATENT:
                prefix_embeddings = None
            else:
                maximum_latent = maximum_latents[branch.student_id]
                controlled_latent = self.controller(maximum_latent, branch.control)
                if controlled_latent is None:
                    raise RuntimeError("latent controller returned no latent unexpectedly")
                prefix_embeddings = self.readers[branch.student_id](controlled_latent, branch.width)
            branch_losses[branch.key] = self.students[branch.student_id].compute_loss(
                prefix_embeddings, student_batches[branch.student_id]
            )

        total_loss = aggregate_weighted_losses(branches, branch_losses)
        return CopeForwardOutput(total_loss, branch_losses, maximum_latents)
