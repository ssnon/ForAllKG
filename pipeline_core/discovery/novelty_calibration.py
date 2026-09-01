from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


HumanNoveltyLabel = Literal[
    "DIRECTLY_KNOWN",
    "ROUTINE_EXTENSION",
    "UNRESOLVED",
    "POTENTIALLY_NON_OBVIOUS",
]

MachineCalibrationBucket = Literal[
    "DIRECTLY_KNOWN",
    "ROUTINE_EXTENSION",
    "UNRESOLVED",
    "NONOBVIOUSNESS_CANDIDATE",
]

CalibrationSource = Literal[
    "HUMAN_AUDIT",
    "EVIDENCE_STATE_CONTROL",
    "SYNTHETIC_CONTROL",
]


@dataclass(frozen=True)
class NoveltyCalibrationCase:
    case_id: str
    source: CalibrationSource
    description: str

    human_label: HumanNoveltyLabel
    structural_status: str

    rationale: str


def calibration_bucket_from_structural_status(
    structural_status: str,
) -> MachineCalibrationBucket:
    """Convert structural analysis into a conservative calibration bucket.

    A structural leap is NOT itself a scientific novelty judgment.

    INTERACTION_LEAP, MECHANISTIC_LEAP, and
    REGIME_OR_THRESHOLD_LEAP only nominate a claim for subsequent
    non-obviousness adjudication.
    """

    if structural_status == "DIRECTLY_KNOWN":
        return "DIRECTLY_KNOWN"

    if structural_status == "ROUTINE_COMPOSITION":
        return "ROUTINE_EXTENSION"

    if structural_status == "INSUFFICIENT_CLOSURE":
        return "UNRESOLVED"

    if structural_status in {
        "INTERACTION_LEAP",
        "MECHANISTIC_LEAP",
        "REGIME_OR_THRESHOLD_LEAP",
    }:
        return "NONOBVIOUSNESS_CANDIDATE"

    return "UNRESOLVED"


def calibration_direction_is_safe(
    case: NoveltyCalibrationCase,
) -> bool:
    """Check whether the machine disposition is directionally safe.

    For DIRECT/ROUTINE/UNRESOLVED controls we expect exact agreement.

    For a human POTENTIALLY_NON_OBVIOUS case, the machine is only
    expected to nominate it as NONOBVIOUSNESS_CANDIDATE. The machine
    must not manufacture the final human scientific judgment itself.
    """

    bucket = calibration_bucket_from_structural_status(
        case.structural_status
    )

    if case.human_label == "POTENTIALLY_NON_OBVIOUS":
        return bucket == "NONOBVIOUSNESS_CANDIDATE"

    return bucket == case.human_label
