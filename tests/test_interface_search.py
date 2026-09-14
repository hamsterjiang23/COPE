import copy

import pytest
import torch

from cope.interface_search import SearchBridge, conditioned_inputs, numeric_answer


@pytest.mark.parametrize(
    "text,answer",
    [
        ("work 10+20; #### 30", "30"),
        (r"answer: \boxed{1,250.00}", "1250"),
        (r"\boxed{\frac{1}{2}}", "1/2"),
        ("The answer is -3.5.", "-7/2"),
        ("There are 3 and 5", None),
        ("#### 2/0", None),
        ("#### 3\nActually \\boxed{4}", "3"),
    ],
)
def test_numeric_protocol(text, answer):
    assert numeric_answer(text) == answer


@pytest.mark.parametrize("kind", ["embedding", "kv_add", "kv_replace"])
def test_bridge_gradients_cache_and_frozen_student(kind):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(3)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(config).requires_grad_(False).eval()
    original = copy.deepcopy(model.state_dict())
    bridge = SearchBridge(48, config, kind, slots=3, width=16, rank=4)
    bridge_before = copy.deepcopy(bridge.state_dict())
    hidden = torch.randn(1, 6, 48)
    z = bridge.encode(hidden)
    memory = bridge.decode(z, model)
    if kind != "embedding":
        assert memory.layers[0].keys.shape == (1, 2, 3, 8)
        assert memory.layers[1].values.shape == (1, 2, 3, 8)
    inputs = conditioned_inputs(model, bridge, memory, [5, 6, 7], [8, 9])
    if kind == "kv_replace":
        assert inputs["input_ids"].tolist() == [[7, 8, 9]]
        assert inputs["position_ids"].tolist() == [[3, 4, 5]]
    loss = model(**inputs, use_cache=True).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.projector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.reader.parameters())
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
    optimizer.step()
    for family in ("projector.", "reader."):
        assert any(
            not torch.equal(bridge_before[k], v)
            for k, v in bridge.state_dict().items()
            if k.startswith(family)
        )
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
    # Fresh memory on each forward, then append exactly one new cache position.
    with torch.no_grad():
        memory = bridge.decode(bridge.encode(hidden), model)
        inputs = conditioned_inputs(model, bridge, memory, [5, 6, 7])
        result = model(**inputs, use_cache=True)
        length = result.past_key_values.get_seq_length()
        expected = 4 if kind == "kv_replace" else 6
        assert length == expected
        result = model(
            input_ids=torch.tensor([[8]]),
            past_key_values=result.past_key_values,
            position_ids=torch.tensor([[length]]),
            attention_mask=torch.ones(1, length + 1),
            use_cache=True,
        )
        assert result.past_key_values.get_seq_length() == length + 1
