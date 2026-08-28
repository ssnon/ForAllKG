from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Literal

from pipeline_core.discovery.question_task_preservation_policy import (
    TaskPreservationAssessment,
    arbitrate_semantic_conflict,
)
from pipeline_core.discovery.question_task_preservation_shadow import (
    SemanticConflictObservation,
)


@dataclass(frozen=True)
class SemanticConflictPair:
    """
    Pair-level aggregation of repeated Bundle semantic-conflict observations.

    A single incumbent/challenger pair may be observed in multiple Bundle
    selection phases. Arbitration should occur once per scientific pair,
    not once per telemetry event.
    """

    incumbent_id: str
    challenger_id: str

    observation_count: int
    max_semantic_overlap: float
    phases: tuple[str, ...]

    incumbent_bundle_rank: int | None = None
    challenger_candidate_rank: int | None = None


@dataclass(frozen=True)
class TaskPreservationPairProposal:
    incumbent_id: str
    challenger_id: str

    observation_count: int
    max_semantic_overlap: float
    phases: tuple[str, ...]

    incumbent_task_class: str
    challenger_task_class: str

    proposed_action: Literal[
        "KEEP_INCUMBENT",
        "REPLACE_WITH_CHALLENGER",
        "NO_ASSESSMENT",
    ]

    reason: str

    shadow_only: Literal[True] = True
    production_selection_changed: Literal[False] = False


@dataclass(frozen=True)
class TaskPreservationPairShadowReport:
    raw_observation_count: int
    unique_pair_count: int
    assessed_pair_count: int

    replacement_pair_count: int
    keep_pair_count: int
    missing_assessment_pair_count: int

    proposals: tuple[
        TaskPreservationPairProposal,
        ...
    ]

    shadow_only: Literal[True] = True
    production_selection_changed: Literal[False] = False


def _min_optional(
    values: Sequence[int | None],
) -> int | None:
    present = [
        value
        for value in values
        if value is not None
    ]

    return (
        min(present)
        if present
        else None
    )


def aggregate_semantic_conflict_observations(
    observations: Sequence[
        SemanticConflictObservation
    ],
) -> tuple[
    SemanticConflictPair,
    ...
]:
    """
    Deduplicate raw Bundle telemetry by ordered
    (incumbent_id, challenger_id) pair.

    Pair order follows first observation order.
    Phase order also follows first occurrence order.
    """

    grouped: dict[
        tuple[str, str],
        list[SemanticConflictObservation],
    ] = {}

    for observation in observations:
        key = (
            observation.incumbent_id,
            observation.challenger_id,
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            observation
        )

    pairs = []

    for (
        incumbent_id,
        challenger_id,
    ), rows in grouped.items():
        phases = tuple(
            dict.fromkeys(
                row.phase
                for row in rows
            )
        )

        pairs.append(
            SemanticConflictPair(
                incumbent_id=incumbent_id,
                challenger_id=challenger_id,
                observation_count=len(rows),
                max_semantic_overlap=max(
                    row.semantic_overlap
                    for row in rows
                ),
                phases=phases,
                incumbent_bundle_rank=(
                    _min_optional(
                        [
                            row.incumbent_bundle_rank
                            for row in rows
                        ]
                    )
                ),
                challenger_candidate_rank=(
                    _min_optional(
                        [
                            row.challenger_candidate_rank
                            for row in rows
                        ]
                    )
                ),
            )
        )

    return tuple(
        pairs
    )


def evaluate_semantic_conflict_pairs_shadow(
    *,
    observations: Sequence[
        SemanticConflictObservation
    ],
    assessments: Mapping[
        str,
        TaskPreservationAssessment,
    ],
) -> TaskPreservationPairShadowReport:
    """
    Produce at most one task-preservation decision per semantic-conflict pair.

    Missing responsiveness assessments are never inferred.
    """

    pairs = aggregate_semantic_conflict_observations(
        observations
    )

    proposals = []

    assessed = 0
    replacements = 0
    keeps = 0
    missing = 0

    for pair in pairs:
        incumbent = assessments.get(
            pair.incumbent_id
        )

        challenger = assessments.get(
            pair.challenger_id
        )

        if (
            incumbent is None
            or challenger is None
        ):
            missing += 1

            proposals.append(
                TaskPreservationPairProposal(
                    incumbent_id=pair.incumbent_id,
                    challenger_id=pair.challenger_id,
                    observation_count=pair.observation_count,
                    max_semantic_overlap=pair.max_semantic_overlap,
                    phases=pair.phases,
                    incumbent_task_class=(
                        incumbent.task_class
                        if incumbent is not None
                        else "MISSING"
                    ),
                    challenger_task_class=(
                        challenger.task_class
                        if challenger is not None
                        else "MISSING"
                    ),
                    proposed_action="NO_ASSESSMENT",
                    reason="TASK_ASSESSMENT_MISSING",
                )
            )

            continue

        assessed += 1

        arbitration = arbitrate_semantic_conflict(
            incumbent=incumbent,
            challenger=challenger,
        )

        if (
            arbitration.decision
            == "REPLACE_WITH_CHALLENGER"
        ):
            replacements += 1
        else:
            keeps += 1

        proposals.append(
            TaskPreservationPairProposal(
                incumbent_id=pair.incumbent_id,
                challenger_id=pair.challenger_id,
                observation_count=pair.observation_count,
                max_semantic_overlap=pair.max_semantic_overlap,
                phases=pair.phases,
                incumbent_task_class=incumbent.task_class,
                challenger_task_class=challenger.task_class,
                proposed_action=arbitration.decision,
                reason=arbitration.reason,
            )
        )

    return TaskPreservationPairShadowReport(
        raw_observation_count=len(
            observations
        ),
        unique_pair_count=len(
            pairs
        ),
        assessed_pair_count=assessed,
        replacement_pair_count=replacements,
        keep_pair_count=keeps,
        missing_assessment_pair_count=missing,
        proposals=tuple(
            proposals
        ),
    )
