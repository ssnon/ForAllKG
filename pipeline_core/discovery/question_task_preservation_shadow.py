from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from pipeline_core.discovery.question_task_preservation_policy import (
    SemanticConflictArbitration,
    TaskPreservationAssessment,
    arbitrate_semantic_conflict,
)


@dataclass(frozen=True)
class SemanticConflictObservation:
    """
    A semantic-diversity conflict observed by DiscoveryBundle selection.

    This is descriptive only. It does not modify Bundle state.
    """

    incumbent_id: str
    challenger_id: str
    semantic_overlap: float
    phase: str

    incumbent_bundle_rank: int | None = None
    challenger_candidate_rank: int | None = None


@dataclass(frozen=True)
class TaskPreservationShadowProposal:
    """
    Shadow recommendation for one observed semantic conflict.

    proposed_action is informational only.
    """

    incumbent_id: str
    challenger_id: str
    semantic_overlap: float
    phase: str

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
class TaskPreservationShadowReport:
    observations_seen: int
    observations_assessed: int

    replacement_proposal_count: int
    keep_proposal_count: int
    missing_assessment_count: int

    proposals: tuple[
        TaskPreservationShadowProposal,
        ...
    ]

    shadow_only: Literal[True] = True
    production_selection_changed: Literal[False] = False


def _proposal_from_arbitration(
    *,
    observation: SemanticConflictObservation,
    incumbent: TaskPreservationAssessment,
    challenger: TaskPreservationAssessment,
    arbitration: SemanticConflictArbitration,
) -> TaskPreservationShadowProposal:
    return TaskPreservationShadowProposal(
        incumbent_id=observation.incumbent_id,
        challenger_id=observation.challenger_id,
        semantic_overlap=observation.semantic_overlap,
        phase=observation.phase,
        incumbent_task_class=incumbent.task_class,
        challenger_task_class=challenger.task_class,
        proposed_action=arbitration.decision,
        reason=arbitration.reason,
    )


def evaluate_semantic_conflicts_shadow(
    *,
    observations: Sequence[
        SemanticConflictObservation
    ],
    assessments: Mapping[
        str,
        TaskPreservationAssessment,
    ],
) -> TaskPreservationShadowReport:
    """
    Evaluate already-observed Bundle semantic conflicts under the frozen
    task-preservation arbitration contract.

    The function deliberately does not:
    - change Bundle ordering;
    - change semantic thresholds;
    - change quality/reserve gates;
    - infer missing task assessments;
    - call an LLM;
    - mutate production selection.

    Every observation is evaluated independently. Multiple proposals are
    diagnostic evidence, not an executable replacement plan.
    """

    proposals: list[
        TaskPreservationShadowProposal
    ] = []

    assessed = 0
    replacements = 0
    keeps = 0
    missing = 0

    for observation in observations:
        incumbent = assessments.get(
            observation.incumbent_id
        )

        challenger = assessments.get(
            observation.challenger_id
        )

        if (
            incumbent is None
            or challenger is None
        ):
            missing += 1

            proposals.append(
                TaskPreservationShadowProposal(
                    incumbent_id=(
                        observation.incumbent_id
                    ),
                    challenger_id=(
                        observation.challenger_id
                    ),
                    semantic_overlap=(
                        observation.semantic_overlap
                    ),
                    phase=observation.phase,
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

        arbitration = (
            arbitrate_semantic_conflict(
                incumbent=incumbent,
                challenger=challenger,
            )
        )

        proposal = (
            _proposal_from_arbitration(
                observation=observation,
                incumbent=incumbent,
                challenger=challenger,
                arbitration=arbitration,
            )
        )

        proposals.append(
            proposal
        )

        if (
            proposal.proposed_action
            == "REPLACE_WITH_CHALLENGER"
        ):
            replacements += 1
        else:
            keeps += 1

    return TaskPreservationShadowReport(
        observations_seen=len(
            observations
        ),
        observations_assessed=assessed,
        replacement_proposal_count=(
            replacements
        ),
        keep_proposal_count=keeps,
        missing_assessment_count=missing,
        proposals=tuple(
            proposals
        ),
    )
