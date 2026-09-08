import torch

from cope.config import ProtocolConfig
from cope.controls import ControlMode
from cope.objectives import BranchMode, BranchPlanner, BranchSpec
from cope.projectors import OrderedPrefixProjector
from cope.readers import TruncatedLinearReader
from cope.reducers import MaskedMeanReducer
from cope.system import CopeSystem
from cope.training import train_step
from tests.helpers import CountingTeacher, RecordingStudent


def _batches() -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    teacher_inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    student_batches = {
        "small": {"target": torch.randn(2, 4)},
        "large": {"target": torch.randn(2, 7)},
    }
    return teacher_inputs, student_batches


def _shared_system() -> tuple[CopeSystem, CountingTeacher]:
    config = ProtocolConfig(6, 8, (2, 4, 8))
    teacher = CountingTeacher(6)
    students = {"small": RecordingStudent(4), "large": RecordingStudent(7)}
    system = CopeSystem(
        config,
        teacher,
        MaskedMeanReducer(),
        students,
        {
            student_id: TruncatedLinearReader(8, student.hidden_size)
            for student_id, student in students.items()
        },
        shared_projector=OrderedPrefixProjector(6, 8),
    )
    return system, teacher


def test_one_teacher_prefill_is_reused_for_all_students_and_widths() -> None:
    system, teacher = _shared_system()
    branches = BranchPlanner(system.config).build(("small", "large"), BranchMode.ALL_WIDTHS)
    output = system(*_batches(), branches)
    assert teacher.calls == 1
    assert len(output.branch_losses) == 6
    assert output.maximum_latents["small"] is output.maximum_latents["large"]


def test_no_latent_branch_skips_teacher_and_prefix_tokens() -> None:
    system, teacher = _shared_system()
    branches = (BranchSpec("small", 8, control=ControlMode.NO_LATENT),)
    output = system(*_batches(), branches)
    assert teacher.calls == 0
    assert output.maximum_latents == {}
    assert system.students["small"].prefixes[-1] is None


def test_private_projectors_create_student_specific_maximum_latents() -> None:
    config = ProtocolConfig(6, 8, (2, 4, 8))
    teacher = CountingTeacher(6)
    students = {"small": RecordingStudent(4), "large": RecordingStudent(7)}
    system = CopeSystem(
        config,
        teacher,
        MaskedMeanReducer(),
        students,
        {
            student_id: TruncatedLinearReader(8, student.hidden_size)
            for student_id, student in students.items()
        },
        private_projectors={student_id: OrderedPrefixProjector(6, 8) for student_id in students},
    )
    branches = BranchPlanner(config).build(("small", "large"), BranchMode.FULL_ONLY)
    output = system(*_batches(), branches)
    assert system.projector_sharing == "private"
    assert output.maximum_latents["small"] is not output.maximum_latents["large"]


def test_train_step_updates_protocol_but_not_frozen_teacher() -> None:
    system, teacher = _shared_system()
    teacher.requires_grad_(False)
    branches = BranchPlanner(system.config).build(("small",), BranchMode.ALL_WIDTHS)
    before_projector = next(system.shared_projector.parameters()).detach().clone()
    before_teacher = teacher.embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in system.parameters() if parameter.requires_grad], lr=1e-2
    )
    metrics = train_step(system, optimizer, *_batches(), branches, max_grad_norm=1.0)
    after_projector = next(system.shared_projector.parameters()).detach()
    assert metrics.loss > 0
    assert not torch.equal(before_projector, after_projector)
    assert torch.equal(before_teacher, teacher.embedding.weight.detach())
