from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_operationalization_resolver import (
    MeasurementOperationalizationCandidate,
    OperationalizationPairCandidate,
    OperationalizationWitnessResolution,
    OperationalizationWitnessResolutionSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


OperationalizationCriticIssueCode = Literal[
    "TARGET_OBSERVABLE_COVERAGE_UNRESOLVED",
    "ANCHOR_METRIC_HETEROGENEITY",
    "TARGET_ANCHOR_METRIC_COLLAPSE",
    "CO_MEASUREMENT_CONTEXT_UNRESOLVED",
]


class OperationalizationCriticIssue(StrictModel):
    code: OperationalizationCriticIssueCode
    message: str
    metrics: dict[str, float] = Field(default_factory=dict)


class OperationalizationResolutionCritique(StrictModel):
    requirement_id: str
    output_experiment_id: str
    resolver_status: str

    target_best_lexical_coverage: float
    anchor_dominant_metric_fraction: float
    same_metric_pair_fraction: float
    cross_paper_pair_fraction: float
    same_paper_distinct_metric_pair_count: int = Field(ge=0)

    issues: list[OperationalizationCriticIssue] = Field(default_factory=list)

    measurement_independence_verified: Literal[False] = False
    independence_certification_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class OperationalizationWitnessCriticReport(StrictModel):
    schema_version: Literal[
        "operationalization-witness-critic-v1"
    ] = "operationalization-witness-critic-v1"

    resolution_count: int = Field(ge=0)
    flagged_resolution_count: int = Field(ge=0)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    resolutions: list[OperationalizationResolutionCritique] = Field(
        default_factory=list
    )

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    measurement_independence_verified_count: Literal[0] = 0
    independence_certification_authority: Literal[False] = False
    experiment_selection_performed: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


def _dominant_metric_fraction(
    rows: list[MeasurementOperationalizationCandidate],
) -> float:
    if not rows:
        return 0.0
    counts = Counter(
        str(row.metric_id or "").strip()
        for row in rows
    )
    return float(max(counts.values()) / len(rows))


def _candidate_lookup(
    resolution: OperationalizationWitnessResolution,
) -> dict[str, MeasurementOperationalizationCandidate]:
    return {
        row.candidate_id: row
        for row in (
            *resolution.target_candidates,
            *resolution.anchor_candidates,
        )
    }


def _shared_source_paper(
    target: MeasurementOperationalizationCandidate,
    anchor: MeasurementOperationalizationCandidate,
) -> bool:
    return bool(
        set(target.source_paper_ids)
        & set(anchor.source_paper_ids)
    )


def _same_metric(
    target: MeasurementOperationalizationCandidate,
    anchor: MeasurementOperationalizationCandidate,
) -> bool:
    left = str(target.metric_id or "").strip()
    right = str(anchor.metric_id or "").strip()
    return bool(left and right and left == right)


def critique_operationalization_resolution(
    resolution: OperationalizationWitnessResolution,
) -> OperationalizationResolutionCritique:
    target_best_coverage = max(
        (
            row.lexical_coverage
            for row in resolution.target_candidates
        ),
        default=0.0,
    )
    anchor_dominant_fraction = _dominant_metric_fraction(
        resolution.anchor_candidates
    )

    lookup = _candidate_lookup(resolution)

    same_metric_count = 0
    cross_paper_count = 0
    same_paper_distinct_metric_count = 0

    for pair in resolution.pair_candidates:
        target = lookup[pair.target_candidate_id]
        anchor = lookup[pair.anchor_candidate_id]

        same_metric = _same_metric(target, anchor)
        shared_paper = _shared_source_paper(target, anchor)

        if same_metric:
            same_metric_count += 1
        if not shared_paper:
            cross_paper_count += 1
        if shared_paper and not same_metric:
            same_paper_distinct_metric_count += 1

    pair_count = len(resolution.pair_candidates)
    same_metric_fraction = (
        float(same_metric_count / pair_count)
        if pair_count
        else 0.0
    )
    cross_paper_fraction = (
        float(cross_paper_count / pair_count)
        if pair_count
        else 0.0
    )

    issues: list[OperationalizationCriticIssue] = []

    if target_best_coverage < 1.0:
        issues.append(
            OperationalizationCriticIssue(
                code="TARGET_OBSERVABLE_COVERAGE_UNRESOLVED",
                message=(
                    "No retrieved target candidate fully covers the requested "
                    "target wording under the current deterministic lexical "
                    "representation. This does not prove scientific mismatch, "
                    "but the target operationalization remains partial."
                ),
                metrics={
                    "target_best_lexical_coverage":
                        target_best_coverage,
                },
            )
        )

    if (
        resolution.anchor_candidates
        and anchor_dominant_fraction < 0.90
    ):
        issues.append(
            OperationalizationCriticIssue(
                code="ANCHOR_METRIC_HETEROGENEITY",
                message=(
                    "The anchor candidate pool is materially heterogeneous "
                    "in canonical metric_id, so lexical retrieval has mixed "
                    "multiple operational quantities."
                ),
                metrics={
                    "anchor_dominant_metric_fraction":
                        anchor_dominant_fraction,
                },
            )
        )

    if same_metric_fraction >= 0.50:
        issues.append(
            OperationalizationCriticIssue(
                code="TARGET_ANCHOR_METRIC_COLLAPSE",
                message=(
                    "At least half of the proposed target/anchor pairs use "
                    "the same non-empty canonical metric_id. Different "
                    "providers or methods do not make the two observables "
                    "operationally independent when the measured quantity "
                    "itself collapses to the same metric."
                ),
                metrics={
                    "same_metric_pair_fraction":
                        same_metric_fraction,
                },
            )
        )

    if (
        resolution.pair_candidates
        and same_paper_distinct_metric_count == 0
    ):
        issues.append(
            OperationalizationCriticIssue(
                code="CO_MEASUREMENT_CONTEXT_UNRESOLVED",
                message=(
                    "No retrieved pair demonstrates two distinct metric_ids "
                    "within a shared source-paper context. Cross-paper method "
                    "diversity alone is not evidence that both observables can "
                    "be independently measured or disentangled in the same "
                    "experimental system."
                ),
                metrics={
                    "cross_paper_pair_fraction":
                        cross_paper_fraction,
                    "same_paper_distinct_metric_pair_count": 0.0,
                },
            )
        )

    return OperationalizationResolutionCritique(
        requirement_id=resolution.requirement_id,
        output_experiment_id=resolution.output_experiment_id,
        resolver_status=resolution.status,
        target_best_lexical_coverage=target_best_coverage,
        anchor_dominant_metric_fraction=anchor_dominant_fraction,
        same_metric_pair_fraction=same_metric_fraction,
        cross_paper_pair_fraction=cross_paper_fraction,
        same_paper_distinct_metric_pair_count=(
            same_paper_distinct_metric_count
        ),
        issues=issues,
    )


def critique_operationalization_witnesses(
    resolutions: OperationalizationWitnessResolutionSet,
) -> OperationalizationWitnessCriticReport:
    rows = [
        critique_operationalization_resolution(row)
        for row in resolutions.resolutions
    ]
    issue_counts = Counter(
        issue.code
        for row in rows
        for issue in row.issues
    )

    return OperationalizationWitnessCriticReport(
        resolution_count=len(rows),
        flagged_resolution_count=sum(
            bool(row.issues)
            for row in rows
        ),
        issue_counts=dict(sorted(issue_counts.items())),
        resolutions=rows,
    )
