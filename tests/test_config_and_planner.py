import pytest
import torch

from cope.config import ProtocolConfig
from cope.controls import ControlMode
from cope.objectives import BranchMode, BranchPlanner


def test_protocol_config_requires_ordered_widths_ending_at_maximum() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ProtocolConfig(6, 8, (4, 2, 8))
    with pytest.raises(ValueError, match="end at max_width"):
        ProtocolConfig(6, 8, (2, 4))


def test_all_width_planner_matches_mrl_objective_grid() -> None:
    config = ProtocolConfig(6, 8, (2, 4, 8))
    branches = BranchPlanner(config).build(("small", "large"), BranchMode.ALL_WIDTHS)
    assert [(branch.student_id, branch.width) for branch in branches] == [
        ("small", 2),
        ("small", 4),
        ("small", 8),
        ("large", 2),
        ("large", 4),
        ("large", 8),
    ]


def test_full_only_planner_uses_maximum_width() -> None:
    config = ProtocolConfig(6, 8, (2, 4, 8))
    branches = BranchPlanner(config).build(
        ("small",), BranchMode.FULL_ONLY, control=ControlMode.NO_LATENT
    )
    assert branches[0].width == 8
    assert branches[0].control is ControlMode.NO_LATENT


def test_sampled_width_applies_inverse_probability_correction() -> None:
    config = ProtocolConfig(6, 8, (2, 4, 8))
    generator = torch.Generator().manual_seed(4)
    branch = BranchPlanner(config).build(
        ("small",),
        BranchMode.SAMPLE_WIDTH,
        weights={("small", 2): 0.5, ("small", 4): 0.5, ("small", 8): 0.5},
        generator=generator,
    )[0]
    assert branch.width in config.widths
    assert branch.weight == 1.5
