"""Named configurations for the decisive COPE factor experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from cope.controls import ControlMode
from cope.objectives import BranchMode


class ExperimentArm(str, Enum):
    PRIVATE_FULL_ONLY = "private_full_only"
    PRIVATE_MULTI_WIDTH = "private_multi_width"
    SHARED_FULL_ONLY = "shared_full_only"
    SHARED_MULTI_WIDTH = "shared_multi_width"
    NO_LATENT = "no_latent"
    ZERO_LATENT = "zero_latent"
    CONSTANT_LATENT = "constant_latent"
    MISMATCHED_LATENT = "mismatched_latent"


@dataclass(frozen=True)
class ArmSettings:
    arm: ExperimentArm
    projector_sharing: str
    branch_mode: BranchMode
    control: ControlMode


def settings_for_arm(arm: ExperimentArm) -> ArmSettings:
    mapping = {
        ExperimentArm.PRIVATE_FULL_ONLY: ArmSettings(
            ExperimentArm.PRIVATE_FULL_ONLY,
            "private",
            BranchMode.FULL_ONLY,
            ControlMode.CORRECT,
        ),
        ExperimentArm.PRIVATE_MULTI_WIDTH: ArmSettings(
            ExperimentArm.PRIVATE_MULTI_WIDTH,
            "private",
            BranchMode.ALL_WIDTHS,
            ControlMode.CORRECT,
        ),
        ExperimentArm.SHARED_FULL_ONLY: ArmSettings(
            ExperimentArm.SHARED_FULL_ONLY,
            "shared",
            BranchMode.FULL_ONLY,
            ControlMode.CORRECT,
        ),
        ExperimentArm.SHARED_MULTI_WIDTH: ArmSettings(
            ExperimentArm.SHARED_MULTI_WIDTH,
            "shared",
            BranchMode.ALL_WIDTHS,
            ControlMode.CORRECT,
        ),
        ExperimentArm.NO_LATENT: ArmSettings(
            ExperimentArm.NO_LATENT,
            "shared",
            BranchMode.FULL_ONLY,
            ControlMode.NO_LATENT,
        ),
        ExperimentArm.ZERO_LATENT: ArmSettings(
            ExperimentArm.ZERO_LATENT,
            "shared",
            BranchMode.ALL_WIDTHS,
            ControlMode.ZERO,
        ),
        ExperimentArm.CONSTANT_LATENT: ArmSettings(
            ExperimentArm.CONSTANT_LATENT,
            "shared",
            BranchMode.ALL_WIDTHS,
            ControlMode.CONSTANT,
        ),
        ExperimentArm.MISMATCHED_LATENT: ArmSettings(
            ExperimentArm.MISMATCHED_LATENT,
            "shared",
            BranchMode.ALL_WIDTHS,
            ControlMode.MISMATCHED,
        ),
    }
    return mapping[arm]


def load_arm_settings(path: str | Path) -> ArmSettings:
    """Load and cross-check a checked-in experiment-arm manifest."""

    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    arm = ExperimentArm(data["arm"])
    expected = settings_for_arm(arm)
    actual = ArmSettings(
        arm=arm,
        projector_sharing=str(data["projector_sharing"]),
        branch_mode=BranchMode(data["branch_mode"]),
        control=ControlMode(data["control"]),
    )
    if actual != expected:
        raise ValueError(f"arm manifest {path} does not match canonical settings for {arm.value}")
    return actual
