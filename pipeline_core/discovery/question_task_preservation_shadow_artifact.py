from __future__ import annotations

from typing import Any, Mapping

from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisTwoPassStability,
)
from pipeline_core.discovery.question_task_preservation_policy import (
    TaskPreservationAssessment,
    classify_task_preservation,
)
from pipeline_core.discovery.question_task_preservation_shadow import (
    SemanticConflictObservation,
)
from pipeline_core.discovery.question_task_preservation_shadow_pairs import (
    evaluate_semantic_conflict_pairs_shadow,
)


def _to_stability(
    row: Mapping[str, Any],
) -> QuestionAxisTwoPassStability:
    return QuestionAxisTwoPassStability(
        pass_1_status=row["pass_1_status"],
        pass_2_status=row["pass_2_status"],
        pass_1_role=row["pass_1_role"],
        pass_2_role=row["pass_2_role"],
        decision_stable=bool(
            row["decision_stable"]
        ),
        stable_status=row.get(
            "stable_status"
        ),
        stable_role=row.get(
            "stable_role"
        ),
    )


def assessments_from_responsiveness_audit(
    *,
    payload: Mapping[str, Any],
    group: str,
) -> dict[
    str,
    TaskPreservationAssessment,
]:
    """
    Convert the existing candidate-level responsiveness audit into the
    frozen coarse task-preservation assessment contract.

    Quality eligibility is intentionally True here because these
    assessments are joined only to semantic-conflict observations emitted
    from DiscoveryBundle's post-quality-gate selection loop.
    """

    rows = payload.get(
        "candidate_results",
        [],
    )

    if not isinstance(rows, list):
        raise ValueError(
            "responsiveness candidate_results must be a list"
        )

    assessments = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        if str(
            row.get("group", "")
        ) != group:
            continue

        unit_id = str(
            row.get("unit_id", "")
        ).strip()

        if not unit_id:
            continue

        assessments[unit_id] = (
            classify_task_preservation(
                candidate_id=unit_id,
                quality_eligible=True,
                stability=_to_stability(
                    row
                ),
            )
        )

    return assessments


def observations_from_raw_conflict_artifact(
    payload: Mapping[str, Any],
) -> list[
    SemanticConflictObservation
]:
    rows = payload.get(
        "observations",
        [],
    )

    if not isinstance(rows, list):
        raise ValueError(
            "raw conflict observations must be a list"
        )

    observations = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        incumbent_id = str(
            row.get(
                "incumbent_id",
                "",
            )
        ).strip()

        challenger_id = str(
            row.get(
                "challenger_id",
                "",
            )
        ).strip()

        if (
            not incumbent_id
            or not challenger_id
        ):
            continue

        observations.append(
            SemanticConflictObservation(
                incumbent_id=incumbent_id,
                challenger_id=challenger_id,
                semantic_overlap=float(
                    row.get(
                        "semantic_overlap",
                        0.0,
                    )
                ),
                phase=str(
                    row.get(
                        "phase",
                        "",
                    )
                ),
                incumbent_bundle_rank=(
                    row.get(
                        "incumbent_bundle_rank"
                    )
                ),
                challenger_candidate_rank=(
                    row.get(
                        "challenger_candidate_rank"
                    )
                ),
            )
        )

    return observations


def build_task_preservation_shadow_artifact(
    *,
    raw_conflict_payload: Mapping[
        str,
        Any,
    ],
    responsiveness_payload: Mapping[
        str,
        Any,
    ],
    group: str,
    raw_conflict_source: str | None = None,
    responsiveness_source: str | None = None,
) -> dict[str, Any]:
    observations = (
        observations_from_raw_conflict_artifact(
            raw_conflict_payload
        )
    )

    assessments = (
        assessments_from_responsiveness_audit(
            payload=responsiveness_payload,
            group=group,
        )
    )

    report = (
        evaluate_semantic_conflict_pairs_shadow(
            observations=observations,
            assessments=assessments,
        )
    )

    return {
        "schema_version":
            "question-task-preservation-pair-shadow-v1",

        "group":
            group,

        "source_raw_conflict_artifact":
            raw_conflict_source,

        "source_responsiveness_artifact":
            responsiveness_source,

        "raw_observation_count":
            report.raw_observation_count,

        "unique_pair_count":
            report.unique_pair_count,

        "assessment_count":
            len(
                assessments
            ),

        "assessed_pair_count":
            report.assessed_pair_count,

        "replacement_pair_count":
            report.replacement_pair_count,

        "keep_pair_count":
            report.keep_pair_count,

        "missing_assessment_pair_count":
            report.missing_assessment_pair_count,

        "proposals": [
            {
                "incumbent_id":
                    proposal.incumbent_id,

                "challenger_id":
                    proposal.challenger_id,

                "observation_count":
                    proposal.observation_count,

                "max_semantic_overlap":
                    proposal.max_semantic_overlap,

                "phases":
                    list(
                        proposal.phases
                    ),

                "incumbent_task_class":
                    proposal.incumbent_task_class,

                "challenger_task_class":
                    proposal.challenger_task_class,

                "proposed_action":
                    proposal.proposed_action,

                "reason":
                    proposal.reason,
            }
            for proposal in report.proposals
        ],

        "shadow_only":
            True,

        "production_selection_changed":
            False,
    }
