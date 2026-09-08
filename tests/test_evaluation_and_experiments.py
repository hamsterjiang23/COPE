from pathlib import Path

from cope.evaluation import (
    QualityWidthRecord,
    correction_statistics,
    latent_payload_bytes,
    pareto_frontier,
)
from cope.experiments import ExperimentArm, load_arm_settings


def test_checked_in_arm_manifests_match_canonical_settings() -> None:
    paths = sorted(Path("configs/arms").glob("*.json"))
    assert len(paths) == len(ExperimentArm)
    assert {load_arm_settings(path).arm for path in paths} == set(ExperimentArm)


def test_payload_size_excludes_protocol_overhead() -> None:
    assert latent_payload_bytes(num_slots=4, width=32, bytes_per_element=2) == 256


def test_pareto_frontier_is_computed_per_student() -> None:
    records = [
        QualityWidthRecord("small", 2, 0.5, 10.0, 4),
        QualityWidthRecord("small", 4, 0.7, 20.0, 8),
        QualityWidthRecord("small", 8, 0.6, 30.0, 16),
        QualityWidthRecord("large", 2, 0.8, 15.0, 4),
    ]
    assert pareto_frontier(records) == [records[0], records[1], records[3]]


def test_correction_statistics_reports_benefit_and_harm() -> None:
    stats = correction_statistics(
        teacher_correct=[True, True, False, True],
        student_correct=[False, True, True, False],
        assisted_correct=[True, False, True, False],
    )
    assert stats.teacher_correct_student_wrong == 2
    assert stats.corrected == 1
    assert stats.correction_rate == 0.5
    assert stats.harmed == 1
    assert stats.harmful_override_rate == 0.5
    assert stats.net_accuracy_change == 0.0
