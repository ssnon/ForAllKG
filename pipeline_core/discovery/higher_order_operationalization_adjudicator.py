from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_operationalization_critic import (
    OperationalizationWitnessCriticReport,
)
from pipeline_core.discovery.higher_order_operationalization_resolver import (
    OperationalizationWitnessResolutionSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


OperationalizationNextRoute = Literal[
    "TARGET_OPERATIONALIZATION_BRIDGE_REQUIRED",
    "ANCHOR_METRIC_DISAMBIGUATION_REQUIRED",
    "TARGET_ANCHOR_METRIC_DISENTANGLEMENT_REQUIRED",
    "CO_MEASUREMENT_WITNESS_REQUIRED",
]

CandidateSupportState = Literal[
    "CO_MEASUREMENT_DISTINCT_METRIC_CANDIDATE_PRESENT",
    "CROSS_PAPER_DISTINCT_ONLY",
    "NO_DISTINCT_OPERATIONALIZATION_CANDIDATE",
]


class OperationalizationWitnessAdjudication(StrictModel):
    requirement_id: str
    output_experiment_id: str

    raw_resolver_status: str
    candidate_support_state: CandidateSupportState
    raw_supported_status_demoted: bool

    target_best_lexical_coverage: float
    anchor_dominant_metric_fraction: float
    same_metric_pair_fraction: float
    cross_paper_pair_fraction: float
    same_paper_distinct_metric_pair_count: int = Field(ge=0)

    next_routes: list[OperationalizationNextRoute] = Field(
        default_factory=list
    )
    issue_codes: list[str] = Field(default_factory=list)

    candidate_support_is_certification: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    independence_certification_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class OperationalizationWitnessAdjudicationSet(StrictModel):
    schema_version: Literal[
        "operationalization-witness-adjudication-v1"
    ] = "operationalization-witness-adjudication-v1"

    resolution_count: int = Field(ge=0)
    adjudication_count: int = Field(ge=0)

    raw_supported_resolution_count: int = Field(ge=0)
    demoted_raw_supported_resolution_count: int = Field(ge=0)
    co_measurement_candidate_resolution_count: int = Field(ge=0)
    cross_paper_only_resolution_count: int = Field(ge=0)

    route_counts: dict[str, int] = Field(default_factory=dict)
    adjudications: list[OperationalizationWitnessAdjudication] = Field(
        default_factory=list
    )

    diagnostic_only: Literal[True] = True
    resolver_retrieval_mutated: Literal[False] = False
    measurement_independence_verified_count: Literal[0] = 0
    independence_certification_authority: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    experiment_selection_performed: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


_ROUTE_BY_ISSUE = {
    "TARGET_OBSERVABLE_COVERAGE_UNRESOLVED":
        "TARGET_OPERATIONALIZATION_BRIDGE_REQUIRED",
    "ANCHOR_METRIC_HETEROGENEITY":
        "ANCHOR_METRIC_DISAMBIGUATION_REQUIRED",
    "TARGET_ANCHOR_METRIC_COLLAPSE":
        "TARGET_ANCHOR_METRIC_DISENTANGLEMENT_REQUIRED",
    "CO_MEASUREMENT_CONTEXT_UNRESOLVED":
        "CO_MEASUREMENT_WITNESS_REQUIRED",
}


def adjudicate_operationalization_witnesses(
    *,
    resolutions: OperationalizationWitnessResolutionSet,
    critique: OperationalizationWitnessCriticReport,
) -> OperationalizationWitnessAdjudicationSet:
    resolution_by_requirement = {
        row.requirement_id: row
        for row in resolutions.resolutions
    }
    critique_by_requirement = {
        row.requirement_id: row
        for row in critique.resolutions
    }

    if set(resolution_by_requirement) != set(critique_by_requirement):
        raise ValueError(
            "operationalization resolution/critique requirement sets differ"
        )

    rows = []

    for requirement_id in sorted(resolution_by_requirement):
        resolution = resolution_by_requirement[requirement_id]
        row_critique = critique_by_requirement[requirement_id]

        if row_critique.same_paper_distinct_metric_pair_count > 0:
            support_state: CandidateSupportState = (
                "CO_MEASUREMENT_DISTINCT_METRIC_CANDIDATE_PRESENT"
            )
        elif resolution.pair_candidate_count > 0:
            support_state = "CROSS_PAPER_DISTINCT_ONLY"
        else:
            support_state = "NO_DISTINCT_OPERATIONALIZATION_CANDIDATE"

        issue_codes = [
            issue.code
            for issue in row_critique.issues
        ]

        routes = []
        for issue_code in issue_codes:
            route = _ROUTE_BY_ISSUE.get(issue_code)
            if route is not None and route not in routes:
                routes.append(route)

        raw_supported = (
            resolution.status == "SUPPORTED_DISTINCT_OPERATIONALIZATION"
        )
        raw_supported_status_demoted = bool(
            raw_supported
            and support_state
            != "CO_MEASUREMENT_DISTINCT_METRIC_CANDIDATE_PRESENT"
        )

        rows.append(
            OperationalizationWitnessAdjudication(
                requirement_id=requirement_id,
                output_experiment_id=resolution.output_experiment_id,
                raw_resolver_status=resolution.status,
                candidate_support_state=support_state,
                raw_supported_status_demoted=raw_supported_status_demoted,
                target_best_lexical_coverage=(
                    row_critique.target_best_lexical_coverage
                ),
                anchor_dominant_metric_fraction=(
                    row_critique.anchor_dominant_metric_fraction
                ),
                same_metric_pair_fraction=(
                    row_critique.same_metric_pair_fraction
                ),
                cross_paper_pair_fraction=(
                    row_critique.cross_paper_pair_fraction
                ),
                same_paper_distinct_metric_pair_count=(
                    row_critique.same_paper_distinct_metric_pair_count
                ),
                next_routes=routes,
                issue_codes=issue_codes,
            )
        )

    route_counts: dict[str, int] = {}
    for row in rows:
        for route in row.next_routes:
            route_counts[route] = route_counts.get(route, 0) + 1

    return OperationalizationWitnessAdjudicationSet(
        resolution_count=resolutions.resolution_count,
        adjudication_count=len(rows),
        raw_supported_resolution_count=sum(
            row.raw_resolver_status
            == "SUPPORTED_DISTINCT_OPERATIONALIZATION"
            for row in rows
        ),
        demoted_raw_supported_resolution_count=sum(
            row.raw_supported_status_demoted
            for row in rows
        ),
        co_measurement_candidate_resolution_count=sum(
            row.candidate_support_state
            == "CO_MEASUREMENT_DISTINCT_METRIC_CANDIDATE_PRESENT"
            for row in rows
        ),
        cross_paper_only_resolution_count=sum(
            row.candidate_support_state
            == "CROSS_PAPER_DISTINCT_ONLY"
            for row in rows
        ),
        route_counts=dict(sorted(route_counts.items())),
        adjudications=rows,
    )
