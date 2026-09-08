import pytest
import torch
import torch.nn.functional as F

from cope.controls import ControlMode, LatentController
from cope.projectors import OrderedPrefixProjector
from cope.readers import TruncatedLinearReader, WidthSpecificReader
from cope.reducers import LearnedQueryReducer, MaskedMeanReducer


def test_ordered_projector_prefix_is_exact_leading_slice() -> None:
    projector = OrderedPrefixProjector(6, 8)
    maximum = projector(torch.randn(2, 3, 6))
    assert maximum.shape == (2, 3, 8)
    assert torch.equal(projector.prefix(maximum, 4), maximum[..., :4])


def test_truncated_reader_matches_manual_linear_operation() -> None:
    reader = TruncatedLinearReader(8, 5)
    latent = torch.randn(2, 3, 8)
    actual = reader(latent, 4)
    expected = F.linear(latent[..., :4], reader.weight[:, :4], reader.bias)
    assert torch.allclose(actual, expected)


def test_width_specific_reader_rejects_unconfigured_width() -> None:
    reader = WidthSpecificReader((2, 4, 8), 5)
    with pytest.raises(ValueError, match="not configured"):
        reader(torch.randn(2, 1, 8), 3)


def test_masked_mean_reducer_ignores_padding() -> None:
    reducer = MaskedMeanReducer(num_slots=1)
    states = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(reducer(states, mask), torch.tensor([[[2.0]]]))


def test_learned_query_reducer_has_fixed_slot_shape() -> None:
    reducer = LearnedQueryReducer(hidden_size=6, num_slots=3, num_heads=2)
    output = reducer(torch.randn(2, 5, 6), torch.ones(2, 5))
    assert output.shape == (2, 3, 6)


def test_latent_controls_preserve_contract_and_mismatch_batch() -> None:
    controller = LatentController(num_slots=1, max_width=4)
    latent = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    assert controller(latent, ControlMode.NO_LATENT) is None
    assert torch.equal(controller(latent, ControlMode.ZERO), torch.zeros_like(latent))
    assert torch.equal(controller(latent, ControlMode.MISMATCHED), latent.roll(1, 0))
    constant = controller(latent, ControlMode.CONSTANT)
    assert constant is not None and constant.shape == latent.shape
