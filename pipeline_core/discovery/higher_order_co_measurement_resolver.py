from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_operationalization_adjudicator import (
    OperationalizationWitnessAdjudicationSet,
)
from pipeline_core.discovery.higher_order_operationalization_resolver import (
    MeasurementOperationalizationCandidate,
    _measurement_candidates,
)
from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirementSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CoMeasurementContextStrength = Literal[
    "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC",
    "SAME_PROVIDER_DISTINCT_METRIC",
    "SAME_SUBJECT_DISTINCT_METRIC",
    "SAME_PAPER_DISTINCT_METRIC",
]

CoMeasurementResolutionStatus = Literal[
    "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC_CANDIDATE",
    "SAME_PROVIDER_DISTINCT_METRIC_CANDIDATE",
    "SAME_SUBJECT_DISTINCT_METRIC_CANDIDATE",
    "SAME_PAPER_DISTINCT_METRIC_CANDIDATE",
    "NO_CO_MEASUREMENT_CANDIDATE",
    "NOT_ROUTED_FOR_CO_MEASUREMENT",
]


class CoMeasurementContextCandidate(StrictModel):
    schema_version: Literal[
        "co-measurement-context-candidate-v1"
    ] = "co-measurement-context-candidate-v1"

    requirement_id: str
    output_experiment_id: str
    source_paper_id: str

    target_candidate_id: str
    anchor_candidate_id: str
    target_measurement_node_id: str
    anchor_measurement_node_id: str

    target_metric_id: str
    anchor_metric_id: str
    distinct_metric_ids: Literal[True] = True

    target_subject_id: str
    anchor_subject_id: str
    same_subject_id: bool

    target_provider_ids: list[str] = Field(default_factory=list)
    anchor_provider_ids: list[str] = Field(default_factory=list)
    shared_provider_ids: list[str] = Field(default_factory=list)
    shared_provider_present: bool

    target_lexical_coverage: float
    anchor_lexical_coverage: float
    context_strength: CoMeasurementContextStrength

    candidate_inspiration_involved: bool = False

    candidate_is_independence_witness: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    independence_certification_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class CoMeasurementRequirementResolution(StrictModel):
    requirement_id: str
    output_experiment_id: str
    status: CoMeasurementResolutionStatus

    target_candidate_count: int = Field(ge=0)
    anchor_candidate_count: int = Field(ge=0)
    shared_source_paper_count: int = Field(ge=0)

    distinct_metric_pair_count: int = Field(ge=0)
    same_subject_pair_count: int = Field(ge=0)
    shared_provider_pair_count: int = Field(ge=0)
    same_provider_and_subject_pair_count: int = Field(ge=0)

    context_strength_counts: dict[str, int] = Field(
        default_factory=dict
    )
    candidates: list[CoMeasurementContextCandidate] = Field(
        default_factory=list
    )

    co_measurement_witness_required: bool
    candidate_inspiration_involved: bool = False

    measurement_independence_verified: Literal[False] = False
    independence_certification_authority: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class CoMeasurementWitnessResolutionSet(StrictModel):
    schema_version: Literal[
        "co-measurement-witness-resolution-set-v1"
    ] = "co-measurement-witness-resolution-set-v1"

    requirement_count: int = Field(ge=0)
    routed_requirement_count: int = Field(ge=0)
    resolution_count: int = Field(ge=0)

    status_counts: dict[str, int] = Field(default_factory=dict)
    total_candidate_count: int = Field(ge=0)
    same_subject_candidate_count: int = Field(ge=0)
    shared_provider_candidate_count: int = Field(ge=0)
    same_provider_and_subject_candidate_count: int = Field(ge=0)

    resolutions: list[CoMeasurementRequirementResolution] = Field(
        default_factory=list
    )

    full_evidence_graph_search: Literal[True] = True
    same_paper_required: Literal[True] = True
    distinct_metric_id_required: Literal[True] = True
    candidate_is_independence_witness: Literal[False] = False

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


_STRENGTH_RANK = {
    "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC": 0,
    "SAME_PROVIDER_DISTINCT_METRIC": 1,
    "SAME_SUBJECT_DISTINCT_METRIC": 2,
    "SAME_PAPER_DISTINCT_METRIC": 3,
}


def _provider_ids(
    candidate: MeasurementOperationalizationCandidate,
) -> set[str]:
    return {
        row.provider_id
        for row in candidate.providers
        if str(row.provider_id).strip()
    }


def _same_subject(
    target: MeasurementOperationalizationCandidate,
    anchor: MeasurementOperationalizationCandidate,
) -> bool:
    left = str(target.subject_id or "").strip()
    right = str(anchor.subject_id or "").strip()
    return bool(left and right and left == right)


def _distinct_metric(
    target: MeasurementOperationalizationCandidate,
    anchor: MeasurementOperationalizationCandidate,
) -> bool:
    left = str(target.metric_id or "").strip()
    right = str(anchor.metric_id or "").strip()
    return bool(left and right and left != right)


def _context_strength(
    *,
    same_subject: bool,
    shared_provider: bool,
) -> CoMeasurementContextStrength:
    if same_subject and shared_provider:
        return "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC"
    if shared_provider:
        return "SAME_PROVIDER_DISTINCT_METRIC"
    if same_subject:
        return "SAME_SUBJECT_DISTINCT_METRIC"
    return "SAME_PAPER_DISTINCT_METRIC"


def _status_from_candidates(
    rows: list[CoMeasurementContextCandidate],
) -> CoMeasurementResolutionStatus:
    strengths = {row.context_strength for row in rows}
    if "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC" in strengths:
        return "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC_CANDIDATE"
    if "SAME_PROVIDER_DISTINCT_METRIC" in strengths:
        return "SAME_PROVIDER_DISTINCT_METRIC_CANDIDATE"
    if "SAME_SUBJECT_DISTINCT_METRIC" in strengths:
        return "SAME_SUBJECT_DISTINCT_METRIC_CANDIDATE"
    if "SAME_PAPER_DISTINCT_METRIC" in strengths:
        return "SAME_PAPER_DISTINCT_METRIC_CANDIDATE"
    return "NO_CO_MEASUREMENT_CANDIDATE"


def resolve_co_measurement_witness_candidates(
    *,
    graph: nx.Graph,
    requirements: OperationalizationWitnessRequirementSet,
    adjudication: OperationalizationWitnessAdjudicationSet,
    minimum_coverage: float = 0.50,
    max_measurement_candidates_per_observable: int = 20000,
    max_output_candidates_per_requirement: int = 200,
) -> CoMeasurementWitnessResolutionSet:
    if minimum_coverage <= 0.0 or minimum_coverage > 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if max_measurement_candidates_per_observable < 1:
        raise ValueError(
            "max_measurement_candidates_per_observable must be >= 1"
        )
    if max_output_candidates_per_requirement < 1:
        raise ValueError(
            "max_output_candidates_per_requirement must be >= 1"
        )

    requirement_by_id = {
        row.requirement_id: row
        for row in requirements.requirements
    }
    adjudication_by_id = {
        row.requirement_id: row
        for row in adjudication.adjudications
    }

    if set(requirement_by_id) != set(adjudication_by_id):
        raise ValueError(
            "requirement/adjudication requirement sets differ"
        )

    # Cache retrieval because the same target observable can appear in
    # multiple requirements.
    retrieval_cache: dict[
        tuple[str, str],
        list[MeasurementOperationalizationCandidate],
    ] = {}

    def retrieve(
        observable: str,
        role: Literal["target", "anchor"],
    ) -> list[MeasurementOperationalizationCandidate]:
        key = (observable, role)
        if key not in retrieval_cache:
            retrieval_cache[key] = _measurement_candidates(
                graph,
                observable=observable,
                role=role,
                minimum_coverage=minimum_coverage,
                max_candidates=(
                    max_measurement_candidates_per_observable
                ),
            )
        return retrieval_cache[key]

    resolutions = []

    for requirement_id in sorted(requirement_by_id):
        requirement = requirement_by_id[requirement_id]
        adjudicated = adjudication_by_id[requirement_id]

        routed = (
            "CO_MEASUREMENT_WITNESS_REQUIRED"
            in adjudicated.next_routes
        )

        if not routed:
            resolutions.append(
                CoMeasurementRequirementResolution(
                    requirement_id=requirement_id,
                    output_experiment_id=requirement.output_experiment_id,
                    status="NOT_ROUTED_FOR_CO_MEASUREMENT",
                    target_candidate_count=0,
                    anchor_candidate_count=0,
                    shared_source_paper_count=0,
                    distinct_metric_pair_count=0,
                    same_subject_pair_count=0,
                    shared_provider_pair_count=0,
                    same_provider_and_subject_pair_count=0,
                    context_strength_counts={},
                    candidates=[],
                    co_measurement_witness_required=False,
                    candidate_inspiration_involved=(
                        requirement.candidate_inspiration_involved
                    ),
                )
            )
            continue

        target_candidates = retrieve(
            requirement.requested_target_observable,
            "target",
        )
        anchor_candidates = retrieve(
            requirement.anchor_observable,
            "anchor",
        )

        target_by_paper: dict[
            str,
            list[MeasurementOperationalizationCandidate],
        ] = defaultdict(list)
        anchor_by_paper: dict[
            str,
            list[MeasurementOperationalizationCandidate],
        ] = defaultdict(list)

        for row in target_candidates:
            for paper_id in row.source_paper_ids:
                if paper_id:
                    target_by_paper[paper_id].append(row)

        for row in anchor_candidates:
            for paper_id in row.source_paper_ids:
                if paper_id:
                    anchor_by_paper[paper_id].append(row)

        shared_papers = sorted(
            set(target_by_paper) & set(anchor_by_paper)
        )

        all_rows: list[CoMeasurementContextCandidate] = []

        for paper_id in shared_papers:
            for target in target_by_paper[paper_id]:
                for anchor in anchor_by_paper[paper_id]:
                    if (
                        target.measurement_node_id
                        == anchor.measurement_node_id
                    ):
                        continue
                    if not _distinct_metric(target, anchor):
                        continue

                    target_provider_ids = _provider_ids(target)
                    anchor_provider_ids = _provider_ids(anchor)
                    shared_provider_ids = sorted(
                        target_provider_ids & anchor_provider_ids
                    )
                    same_subject = _same_subject(target, anchor)
                    shared_provider = bool(shared_provider_ids)
                    strength = _context_strength(
                        same_subject=same_subject,
                        shared_provider=shared_provider,
                    )

                    all_rows.append(
                        CoMeasurementContextCandidate(
                            requirement_id=requirement_id,
                            output_experiment_id=(
                                requirement.output_experiment_id
                            ),
                            source_paper_id=paper_id,
                            target_candidate_id=target.candidate_id,
                            anchor_candidate_id=anchor.candidate_id,
                            target_measurement_node_id=(
                                target.measurement_node_id
                            ),
                            anchor_measurement_node_id=(
                                anchor.measurement_node_id
                            ),
                            target_metric_id=str(
                                target.metric_id or ""
                            ),
                            anchor_metric_id=str(
                                anchor.metric_id or ""
                            ),
                            target_subject_id=str(
                                target.subject_id or ""
                            ),
                            anchor_subject_id=str(
                                anchor.subject_id or ""
                            ),
                            same_subject_id=same_subject,
                            target_provider_ids=sorted(
                                target_provider_ids
                            ),
                            anchor_provider_ids=sorted(
                                anchor_provider_ids
                            ),
                            shared_provider_ids=shared_provider_ids,
                            shared_provider_present=shared_provider,
                            target_lexical_coverage=(
                                target.lexical_coverage
                            ),
                            anchor_lexical_coverage=(
                                anchor.lexical_coverage
                            ),
                            context_strength=strength,
                            candidate_inspiration_involved=(
                                requirement
                                .candidate_inspiration_involved
                            ),
                        )
                    )

        # Deduplicate exact measurement-pair/paper realizations.
        deduped = {}
        for row in all_rows:
            key = (
                row.source_paper_id,
                row.target_measurement_node_id,
                row.anchor_measurement_node_id,
            )
            deduped[key] = row

        ranked = list(deduped.values())
        ranked.sort(
            key=lambda row: (
                _STRENGTH_RANK[row.context_strength],
                -row.target_lexical_coverage,
                -row.anchor_lexical_coverage,
                row.source_paper_id,
                row.target_metric_id,
                row.anchor_metric_id,
                row.target_measurement_node_id,
                row.anchor_measurement_node_id,
            )
        )

        strength_counts = Counter(
            row.context_strength
            for row in ranked
        )

        output_rows = ranked[
            :max_output_candidates_per_requirement
        ]

        resolutions.append(
            CoMeasurementRequirementResolution(
                requirement_id=requirement_id,
                output_experiment_id=requirement.output_experiment_id,
                status=_status_from_candidates(ranked),
                target_candidate_count=len(target_candidates),
                anchor_candidate_count=len(anchor_candidates),
                shared_source_paper_count=len(shared_papers),
                distinct_metric_pair_count=len(ranked),
                same_subject_pair_count=sum(
                    row.same_subject_id
                    for row in ranked
                ),
                shared_provider_pair_count=sum(
                    row.shared_provider_present
                    for row in ranked
                ),
                same_provider_and_subject_pair_count=sum(
                    row.same_subject_id
                    and row.shared_provider_present
                    for row in ranked
                ),
                context_strength_counts=dict(
                    sorted(strength_counts.items())
                ),
                candidates=output_rows,
                co_measurement_witness_required=True,
                candidate_inspiration_involved=(
                    requirement.candidate_inspiration_involved
                ),
            )
        )

    status_counts = Counter(
        row.status
        for row in resolutions
    )

    return CoMeasurementWitnessResolutionSet(
        requirement_count=requirements.requirement_count,
        routed_requirement_count=sum(
            row.co_measurement_witness_required
            for row in resolutions
        ),
        resolution_count=len(resolutions),
        status_counts=dict(sorted(status_counts.items())),
        total_candidate_count=sum(
            row.distinct_metric_pair_count
            for row in resolutions
        ),
        same_subject_candidate_count=sum(
            row.same_subject_pair_count
            for row in resolutions
        ),
        shared_provider_candidate_count=sum(
            row.shared_provider_pair_count
            for row in resolutions
        ),
        same_provider_and_subject_candidate_count=sum(
            row.same_provider_and_subject_pair_count
            for row in resolutions
        ),
        resolutions=resolutions,
    )
