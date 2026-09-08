import torch

from cope.adapters import HuggingFaceCausalStudentAdapter, HuggingFaceTeacherAdapter
from tests.helpers import FakeCausalLM, FakeTeacherModel


def test_teacher_adapter_extracts_requested_hidden_state_and_freezes_model() -> None:
    model = FakeTeacherModel(hidden_size=6)
    adapter = HuggingFaceTeacherAdapter(model, hidden_state_index=-1, freeze=True)
    output = adapter({"input_ids": torch.tensor([[1, 2, 3]])})
    assert output.shape == (1, 3, 6)
    assert model.calls == 1
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_student_adapter_prepends_prefix_and_masks_prefix_labels() -> None:
    model = FakeCausalLM(hidden_size=5)
    adapter = HuggingFaceCausalStudentAdapter(model, freeze_backbone=True)
    prefix = torch.randn(2, 3, 5, requires_grad=True)
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    loss = adapter.compute_loss(prefix, batch)
    loss.backward()
    assert model.last_inputs["inputs_embeds"].shape == (2, 5, 5)
    assert model.last_inputs["attention_mask"].shape == (2, 5)
    assert torch.equal(model.last_inputs["labels"][:, :3], torch.full((2, 3), -100))
    assert prefix.grad is not None
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_student_adapter_no_latent_keeps_original_sequence_length() -> None:
    model = FakeCausalLM(hidden_size=5)
    adapter = HuggingFaceCausalStudentAdapter(model)
    batch = {"input_ids": torch.tensor([[1, 2]])}
    adapter.compute_loss(None, batch)
    assert model.last_inputs["inputs_embeds"].shape[1] == 2
