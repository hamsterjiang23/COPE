"""Task-independent diagnostics for COPE experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal


def latent_payload_bytes(num_slots: int, width: int, bytes_per_element: int) -> int:
    """Return the tightly packed latent payload, excluding protocol overhead."""

    if num_slots <= 0 or width <= 0 or bytes_per_element <= 0:
        raise ValueError("num_slots, width, and bytes_per_element must be positive")
    return num_slots * width * bytes_per_element


@dataclass(frozen=True)
class QualityWidthRecord:
    student_id: str
    width: int
    quality: float
    latency_ms: float
    payload_bytes: int

    def __post_init__(self) -> None:
        if not self.student_id:
            raise ValueError("student_id must not be empty")
        if self.width < 0 or self.latency_ms < 0 or self.payload_bytes < 0:
            raise ValueError("width and cost values must be non-negative")
        if not isfinite(self.quality) or not isfinite(self.latency_ms):
            raise ValueError("quality and latency must be finite")


def pareto_frontier(
    records: Sequence[QualityWidthRecord],
    cost: Literal["latency_ms", "payload_bytes"] = "latency_ms",
) -> list[QualityWidthRecord]:
    """Return non-dominated records per student, maximizing quality and minimizing cost."""

    frontier: list[QualityWidthRecord] = []
    for candidate in records:
        candidate_cost = float(getattr(candidate, cost))
        dominated = any(
            other.student_id == candidate.student_id
            and float(getattr(other, cost)) <= candidate_cost
            and other.quality >= candidate.quality
            and (float(getattr(other, cost)) < candidate_cost or other.quality > candidate.quality)
            for other in records
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item.student_id, float(getattr(item, cost))))


@dataclass(frozen=True)
class CorrectionStatistics:
    teacher_correct_student_wrong: int
    corrected: int
    student_correct: int
    harmed: int
    total: int

    @property
    def correction_rate(self) -> float:
        denominator = self.teacher_correct_student_wrong
        return self.corrected / denominator if denominator else 0.0

    @property
    def harmful_override_rate(self) -> float:
        return self.harmed / self.student_correct if self.student_correct else 0.0

    @property
    def net_accuracy_change(self) -> float:
        return (self.corrected - self.harmed) / self.total if self.total else 0.0


def correction_statistics(
    teacher_correct: Sequence[bool],
    student_correct: Sequence[bool],
    assisted_correct: Sequence[bool],
) -> CorrectionStatistics:
    """Measure correction and harmful override rates on the same frozen examples."""

    if not (len(teacher_correct) == len(student_correct) == len(assisted_correct)):
        raise ValueError("all correctness sequences must have the same length")
    teacher_student_gap = sum(
        teacher and not student
        for teacher, student in zip(teacher_correct, student_correct, strict=True)
    )
    corrected = sum(
        teacher and not student and assisted
        for teacher, student, assisted in zip(
            teacher_correct, student_correct, assisted_correct, strict=True
        )
    )
    student_wins = sum(student_correct)
    harmed = sum(
        student and not assisted
        for student, assisted in zip(student_correct, assisted_correct, strict=True)
    )
    return CorrectionStatistics(
        teacher_correct_student_wrong=teacher_student_gap,
        corrected=corrected,
        student_correct=student_wins,
        harmed=harmed,
        total=len(student_correct),
    )
