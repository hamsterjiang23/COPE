"""COPE research framework."""

from cope.config import ProtocolConfig
from cope.controls import ControlMode, LatentController
from cope.evaluation import (
    CorrectionStatistics,
    QualityWidthRecord,
    correction_statistics,
    latent_payload_bytes,
    pareto_frontier,
)
from cope.experiments import ExperimentArm, load_arm_settings, settings_for_arm
from cope.factory import build_huggingface_system
from cope.objectives import BranchMode, BranchPlanner, BranchSpec
from cope.projectors import OrderedPrefixProjector
from cope.readers import TruncatedLinearReader, WidthSpecificReader
from cope.reducers import LearnedQueryReducer, MaskedMeanReducer
from cope.system import CopeForwardOutput, CopeSystem

__all__ = [
    "BranchMode",
    "BranchPlanner",
    "BranchSpec",
    "ControlMode",
    "CorrectionStatistics",
    "CopeForwardOutput",
    "CopeSystem",
    "ExperimentArm",
    "LatentController",
    "LearnedQueryReducer",
    "MaskedMeanReducer",
    "OrderedPrefixProjector",
    "ProtocolConfig",
    "QualityWidthRecord",
    "TruncatedLinearReader",
    "WidthSpecificReader",
    "build_huggingface_system",
    "correction_statistics",
    "latent_payload_bytes",
    "load_arm_settings",
    "pareto_frontier",
    "settings_for_arm",
]
