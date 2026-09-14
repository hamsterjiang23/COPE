import json

import pytest
import torch

from cope.adapters import HuggingFaceCausalStudentAdapter
from cope.experiment_data import (
    donor_indices,
    extract_answer,
    paired_diagnostics,
    prepare_examples,
    write_new_json,
)
from cope.real_experiment import Experiment, assert_frozen_backbones, student_batch, validate_recipe


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [3] + [4] * len(messages[-1]["content"].split())

    def encode(self, text, **kwargs):
        return [5] * len(text.split())


def test_split_is_deterministic_disjoint_and_keeps_long_holdout_answers():
    config = {
        "seed": 42,
        "dataset": {"train_size": 4, "holdout_size": 2},
        "training": {"prompt_max_tokens": 20, "sequence_max_tokens": 40},
    }
    rows = [{"question": f"question {i}", "answer": "reason #### 3"} for i in range(12)]
    rows.append(dict(rows[0]))
    tokenizer = Tokenizer()
    result = prepare_examples(rows, tokenizer, tokenizer, config)
    assert result == prepare_examples(rows, tokenizer, tokenizer, config)
    assert not ({r["id"] for r in result["train"]} & {r["id"] for r in result["holdout"]})
    assert len(result["duplicate_source_indices"]) == 1
    index = result["holdout"][0]["source_index"]
    rows[index]["answer"] = "long " * 1000
    changed = prepare_examples(rows, tokenizer, tokenizer, config)
    assert [r["id"] for r in changed["holdout"]] == [r["id"] for r in result["holdout"]]
    assert all(5 not in r["teacher_ids"] for r in changed["train"])


def test_training_masks_question_and_padding_without_dropping_targets():
    rows = [
        {"prompt_ids": [3, 4], "target_ids": [5, 2]},
        {"prompt_ids": [3], "target_ids": [5, 6, 2]},
    ]
    batch = student_batch(rows, Tokenizer(), "cpu")
    assert batch["labels"].tolist() == [[-100, -100, 5, 2], [-100, 5, 6, 2]]
    rows[1]["target_ids"] = [2]
    batch = student_batch(rows, Tokenizer(), "cpu")
    assert batch["labels"][1].tolist() == [-100, 2, -100, -100]


def test_donors_are_reproducible_and_change_between_epochs():
    first = donor_indices(16, 42, 0)
    assert first == donor_indices(16, 42, 0)
    assert first != donor_indices(16, 42, 1)
    assert sorted(first) == list(range(16))
    assert all(i != donor for i, donor in enumerate(first))
    with pytest.raises(ValueError):
        donor_indices(1, 42)


def test_paired_metrics_include_all_corrections_and_zero_denominators():
    base = [{"id": "a", "correct": False}, {"id": "b", "correct": True}]
    other = [{"id": "a", "correct": True}, {"id": "b", "correct": False}]
    metrics = paired_diagnostics(base, other)
    assert metrics["corrected"] == metrics["harmed"] == 1
    assert metrics["net_accuracy_change"] == 0
    assert paired_diagnostics(other[:1], other[:1])["correction_rate"] is None
    with pytest.raises(ValueError):
        paired_diagnostics(base, other[::-1])
    assert extract_answer("reason\n#### -1,200.50") == "-1200.5"
    assert extract_answer("The answer is 3") is None


def test_evidence_files_cannot_be_overwritten(tmp_path):
    target = tmp_path / "metrics.json"
    write_new_json(target, {"n": 2})
    with pytest.raises(FileExistsError):
        write_new_json(target, {"n": 3})
    assert json.loads(target.read_text()) == {"n": 2}


def test_checkpoint_restores_optimizer_rng_and_weights(tmp_path):
    run = Experiment.__new__(Experiment)
    run.root = tmp_path
    run.latest = {}
    run.config = {"seed": 42, "training": {"interface_lr": 0.01}}
    run.params = {"private_projectors.weight": torch.nn.Parameter(torch.ones(2))}
    run.initial = {"private_projectors.weight": torch.ones(2)}
    run.event = lambda *args, **kwargs: None
    optimizer = run.optimizer("correct")
    run.params["private_projectors.weight"].sum().backward()
    optimizer.step()
    torch.manual_seed(7)
    path = run.checkpoint("correct", 32, optimizer)
    expected = torch.rand(2)
    run.params["private_projectors.weight"].data.zero_()
    recovered, step = run.restore("correct", path)
    assert step == 32
    assert torch.equal(torch.rand(2), expected)
    assert torch.all(run.params["private_projectors.weight"] > 0)
    assert recovered.state_dict()["state"]
    with pytest.raises(FileExistsError):
        run.checkpoint("correct", 32, recovered)
    with pytest.raises(ValueError, match="no optimizer"):
        run.optimizer("no_latent")


def test_new_runner_rejects_lora_recipe_before_loading_models():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="must be frozen"):
        validate_recipe(json.loads((root / "goal/v1/config.json").read_text()))
    validate_recipe(json.loads((root / "goal/v2/config.json").read_text()))


def test_multiwidth_loss_updates_only_interface_with_one_teacher_encoding():
    transformers = pytest.importorskip("transformers")
    from cope.config import ProtocolConfig
    from cope.factory import build_huggingface_system

    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
        eos_token_id=2,
    )
    torch.manual_seed(42)
    teacher = transformers.Qwen3ForCausalLM(config)
    student = transformers.Qwen3ForCausalLM(config)
    run = Experiment.__new__(Experiment)
    run.system = build_huggingface_system(
        ProtocolConfig(32, 16, (8, 16), 2),
        teacher.model,
        {"student": student},
        projector_sharing="private",
        freeze_teacher=True,
        freeze_student_backbones=True,
    )
    run.config = {
        "teacher": {"device": "cpu"},
        "student": {"device": "cpu"},
        "protocol": {"widths": [8, 16], "max_width": 16},
        "training": {"interface_lr": 0.01},
    }
    run.teacher_tokenizer = run.student_tokenizer = Tokenizer()
    run.params = {n: p for n, p in run.system.named_parameters() if p.requires_grad}
    before = {n: p.detach().clone() for n, p in run.system.named_parameters()}
    run.teacher_calls = 0
    run.system.teacher.register_forward_hook(run.count_teacher)
    rows = [{"teacher_ids": [3, 4], "prompt_ids": [3, 4], "target_ids": [5, 2]}]
    run.system.train()
    assert not teacher.model.training and not student.training
    loss = run.loss(rows, "correct")
    assert run.teacher_calls == 1
    loss.backward()
    assert_frozen_backbones(run.system)
    reader_grad = run.system.readers["student"].weight.grad
    assert reader_grad[:, :8].abs().sum() > 0
    assert reader_grad[:, 8:].abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in run.system.private_projectors.parameters()
    )
    run.optimizer("correct").step()
    changed = {n for n, p in run.system.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all(n.startswith(("private_projectors.", "readers.")) for n in changed)
    assert all(
        torch.equal(p, before[n])
        for n, p in run.system.named_parameters()
        if n.startswith(("teacher.", "students."))
    )
    assert not run.loss(rows, "no_latent").requires_grad


def test_budget_check_stops_without_starting_more_work():
    from cope.real_experiment import BudgetExpired

    run = Experiment.__new__(Experiment)
    run.stop_requested = False
    run.deadline = 0
    with pytest.raises(BudgetExpired):
        run.check_budget()


def test_tiny_qwen_padding_loss_and_new_token_generation():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
        eos_token_id=2,
    )
    torch.manual_seed(42)
    model = transformers.Qwen3ForCausalLM(config).eval()
    adapter = HuggingFaceCausalStudentAdapter(model)
    prefix = torch.randn(1, 2, 32)
    single = {
        "input_ids": torch.tensor([[3, 4]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": torch.tensor([[-100, 4]]),
    }
    padded = {
        "input_ids": torch.tensor([[0, 0, 3, 4]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1]]),
        "labels": torch.tensor([[-100, -100, -100, 4]]),
    }
    assert torch.allclose(
        adapter.compute_loss(prefix, single), adapter.compute_loss(prefix, padded), atol=1e-5
    )
    first = adapter.generate(prefix, single, max_new_tokens=3, min_new_tokens=3, do_sample=False)
    second = adapter.generate(prefix, padded, max_new_tokens=3, min_new_tokens=3, do_sample=False)
    assert first.shape == (1, 3)
    assert torch.equal(first, second)
