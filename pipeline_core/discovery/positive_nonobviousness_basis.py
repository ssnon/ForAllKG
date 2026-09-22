from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphReport,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    ScientificHypothesisEvidenceAggregationReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


PositiveBasisKind = Literal[
    "DOCUMENTED_DIRECTIONAL_TENSION",
    "DOCUMENTED_CONTEXTUAL_TENSION",
    "DOCUMENTED_CONFLICTING_PRIOR_EXPECTATION",
]

BasisRoleRelevance = Literal[
    "PRIMARY_NOVELTY_BEARING",
    "REQUIRED_ENABLING",
    "TESTING_ONLY",
    "AUXILIARY_ONLY",
]

ClaimPositiveBasisState = Literal[
    "PRIMARY_POSITIVE_BASIS_CANDIDATE",
    "ENABLING_POSITIVE_BASIS_CANDIDATE",
    "NONQUALIFYING_TENSION_ONLY",
    "NO_POSITIVE_BASIS_CANDIDATE",
]

HypothesisPositiveBasisState = Literal[
    "QUALIFYING_POSITIVE_BASIS_CANDIDATE_PRESENT",
    "NONQUALIFYING_TENSION_ONLY",
    "NO_POSITIVE_BASIS_CANDIDATE",
]

FutureAdjudicationReadiness = Literal[
    "READY_FOR_POSITIVE_NONOBVIOUSNESS_ADJUDICATION",
    "BASIS_PRESENT_BUT_EPISTEMICALLY_PARTIAL",
    "BASIS_PRESENT_BUT_DIRECT_PRIOR_ART_BLOCKS_CERTIFICATION",
    "NONQUALIFYING_TENSION_ONLY",
    "NO_POSITIVE_BASIS",
]

_RELATIONSHIP_TO_BASIS_KIND = {
    "DIRECTIONAL_COUNTEREVIDENCE":
        "DOCUMENTED_DIRECTIONAL_TENSION",
    "CONTEXTUAL_CONFLICT":
        "DOCUMENTED_CONTEXTUAL_TENSION",
    "CONFLICTING_PRIOR_ART":
        "DOCUMENTED_CONFLICTING_PRIOR_EXPECTATION",
}

_ROLE_RELEVANCE = {
    "NOVELTY_BEARING":
        "PRIMARY_NOVELTY_BEARING",
    "REQUIRED_ENABLING_RELATION":
        "REQUIRED_ENABLING",
    "TESTING_PREDICTION":
        "TESTING_ONLY",
    "AUXILIARY":
        "AUXILIARY_ONLY",
}

_QUALIFYING_ROLE_RELEVANCE = frozenset(
    {
        "PRIMARY_NOVELTY_BEARING",
        "REQUIRED_ENABLING",
    }
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


class PositiveNonObviousnessBasisEvidence(StrictModel):
    hypothesis_id: str
    claim_id: str
    work_id: str

    relationship: Literal[
        "DIRECTIONAL_COUNTEREVIDENCE",
        "CONTEXTUAL_CONFLICT",
        "CONFLICTING_PRIOR_ART",
    ]
    basis_kind: PositiveBasisKind

    evidence_span: str = Field(min_length=1)
    rationale: str = ""
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    basis_projection_ids: list[str] = Field(default_factory=list)
    deterministic_reason_codes: list[str] = Field(
        default_factory=list
    )

    # This item establishes only that a prior-art tension/conflict was
    # positively observed in the bounded adjudicated evidence. It does not
    # establish that the proposed hypothesis is true, novel, or non-obvious.
    source_is_adjudicated_relation_evidence: Literal[True] = True
    truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    nonobviousness_authority: Literal[False] = False


class ClaimPositiveNonObviousnessBasisRecord(StrictModel):
    hypothesis_id: str
    claim_id: str

    novelty_selection_role: str
    role_relevance: BasisRoleRelevance
    structural_centrality_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    centrality_rank_within_hypothesis: int = Field(ge=1)
    highest_centrality_within_hypothesis: bool

    classification_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    classification_coverage_state: str

    basis_evidence: list[
        PositiveNonObviousnessBasisEvidence
    ]
    basis_evidence_count: int = Field(ge=0)
    basis_kinds: list[PositiveBasisKind]
    basis_work_ids: list[str]

    direct_prior_art_work_ids: list[str]
    lower_order_prior_art_work_ids: list[str]

    direct_prior_art_blocker_present: bool
    lower_order_prior_art_pressure_present: bool

    basis_state: ClaimPositiveBasisState
    qualifying_for_future_nonobviousness_adjudication: bool

    absence_based_basis_forbidden: Literal[True] = True
    component_only_basis_forbidden: Literal[True] = True
    partial_prior_art_basis_forbidden: Literal[True] = True
    no_strong_pressure_basis_forbidden: Literal[True] = True

    truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_basis(
        self,
    ) -> "ClaimPositiveNonObviousnessBasisRecord":
        if self.basis_evidence_count != len(
            self.basis_evidence
        ):
            raise ValueError(
                "claim positive basis evidence count mismatch"
            )
        if sorted(
            set(row.basis_kind for row in self.basis_evidence)
        ) != sorted(set(self.basis_kinds)):
            raise ValueError(
                "claim positive basis kinds mismatch"
            )
        if sorted(
            set(row.work_id for row in self.basis_evidence)
        ) != sorted(set(self.basis_work_ids)):
            raise ValueError(
                "claim positive basis work IDs mismatch"
            )
        if self.direct_prior_art_blocker_present != bool(
            self.direct_prior_art_work_ids
        ):
            raise ValueError(
                "direct prior-art blocker mismatch"
            )
        if self.lower_order_prior_art_pressure_present != bool(
            self.lower_order_prior_art_work_ids
        ):
            raise ValueError(
                "lower-order pressure mismatch"
            )

        expected_qualifying = bool(
            self.basis_evidence
            and self.role_relevance
            in _QUALIFYING_ROLE_RELEVANCE
        )
        if (
            self.qualifying_for_future_nonobviousness_adjudication
            != expected_qualifying
        ):
            raise ValueError(
                "future non-obviousness eligibility mismatch"
            )
        return self


class HypothesisPositiveNonObviousnessBasisSummary(StrictModel):
    hypothesis_id: str

    claim_ids: list[str]
    qualifying_basis_claim_ids: list[str]
    nonqualifying_tension_claim_ids: list[str]
    no_basis_claim_ids: list[str]

    basis_work_ids: list[str]
    basis_kind_counts: dict[str, int]

    direct_prior_art_blocker_claim_ids: list[str]
    lower_order_pressure_claim_ids: list[str]

    positive_basis_state: HypothesisPositiveBasisState
    future_adjudication_readiness: FutureAdjudicationReadiness

    epistemic_coverage_profile: str
    all_claims_fully_classified: bool
    any_unclassified_presented_work: bool

    # A positive tension/conflict basis is necessary for the next authority
    # layer, but it is not sufficient. It can coexist with evidence that the
    # hypothesis is implausible or already directly anticipated.
    positive_basis_is_necessary_not_sufficient: Literal[True] = True
    direct_prior_art_blocks_future_certification: bool

    absence_based_nonobviousness_forbidden: Literal[True] = True
    truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_partition(
        self,
    ) -> "HypothesisPositiveNonObviousnessBasisSummary":
        claims = set(self.claim_ids)
        partitions = [
            set(self.qualifying_basis_claim_ids),
            set(self.nonqualifying_tension_claim_ids),
            set(self.no_basis_claim_ids),
        ]
        if any(not part.issubset(claims) for part in partitions):
            raise ValueError(
                "positive basis partition contains unknown claim"
            )
        if (
            partitions[0] & partitions[1]
            or partitions[0] & partitions[2]
            or partitions[1] & partitions[2]
        ):
            raise ValueError(
                "positive basis claim partitions overlap"
            )
        if set().union(*partitions) != claims:
            raise ValueError(
                "positive basis claim partitions incomplete"
            )
        if self.direct_prior_art_blocks_future_certification != bool(
            self.direct_prior_art_blocker_claim_ids
        ):
            raise ValueError(
                "hypothesis direct-prior-art blocker mismatch"
            )
        return self


class PositiveNonObviousnessBasisReport(StrictModel):
    schema_version: Literal[
        "positive-nonobviousness-basis-report-v1"
    ] = "positive-nonobviousness-basis-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_claim_evidence_graph_id: str
    source_hypothesis_evidence_aggregation_report_id: str

    claim_records: list[
        ClaimPositiveNonObviousnessBasisRecord
    ]
    hypothesis_summaries: list[
        HypothesisPositiveNonObviousnessBasisSummary
    ]

    claim_count: int = Field(ge=0)
    hypothesis_count: int = Field(ge=0)
    basis_evidence_count: int = Field(ge=0)
    qualifying_basis_claim_count: int = Field(ge=0)

    positive_basis_state_counts: dict[str, int]
    readiness_counts: dict[str, int]

    basis_semantics: Literal[
        "adjudicated_tension_conflict_only_no_absence_inference_v1"
    ] = "adjudicated_tension_conflict_only_no_absence_inference_v1"

    direct_prior_art_is_not_positive_basis: Literal[True] = True
    lower_order_prior_art_is_not_positive_basis: Literal[True] = True
    partial_prior_art_is_not_positive_basis: Literal[True] = True
    component_only_is_not_positive_basis: Literal[True] = True
    retrieval_miss_is_not_positive_basis: Literal[True] = True

    positive_basis_is_not_truth_evidence: Literal[True] = True
    positive_basis_is_not_novelty_verdict: Literal[True] = True
    positive_basis_is_not_nonobviousness_verdict: Literal[True] = True

    positive_nonobviousness_authority_created: Literal[False] = False
    novelty_verdict_created: Literal[False] = False
    certification_performed: Literal[False] = False

    diagnostic_only: Literal[True] = True
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "PositiveNonObviousnessBasisReport":
        if self.claim_count != len(self.claim_records):
            raise ValueError(
                "positive basis claim_count mismatch"
            )
        if self.hypothesis_count != len(
            self.hypothesis_summaries
        ):
            raise ValueError(
                "positive basis hypothesis_count mismatch"
            )
        if self.basis_evidence_count != sum(
            row.basis_evidence_count
            for row in self.claim_records
        ):
            raise ValueError(
                "positive basis evidence_count mismatch"
            )
        if self.qualifying_basis_claim_count != sum(
            row.qualifying_for_future_nonobviousness_adjudication
            for row in self.claim_records
        ):
            raise ValueError(
                "qualifying positive basis claim count mismatch"
            )

        expected_states: dict[str, int] = defaultdict(int)
        expected_readiness: dict[str, int] = defaultdict(int)
        for row in self.hypothesis_summaries:
            expected_states[row.positive_basis_state] += 1
            expected_readiness[
                row.future_adjudication_readiness
            ] += 1

        if dict(sorted(expected_states.items())) != dict(
            sorted(self.positive_basis_state_counts.items())
        ):
            raise ValueError(
                "positive basis state counts mismatch"
            )
        if dict(sorted(expected_readiness.items())) != dict(
            sorted(self.readiness_counts.items())
        ):
            raise ValueError(
                "positive basis readiness counts mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "positive non-obviousness basis SHA mismatch"
            )
        if observed_id != (
            "positive_nonobviousness_basis:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "positive non-obviousness basis ID mismatch"
            )
        return self


def _work_id_by_node(
    graph: ScientificClaimEvidenceGraphReport,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for node in graph.nodes:
        if node.node_kind != "WORK":
            continue
        output[node.node_id] = (
            node.source_ids[0]
            if node.source_ids
            else node.node_id
        )
    return output


def _basis_state(
    *,
    role_relevance: BasisRoleRelevance,
    evidence_count: int,
) -> ClaimPositiveBasisState:
    if evidence_count == 0:
        return "NO_POSITIVE_BASIS_CANDIDATE"
    if role_relevance == "PRIMARY_NOVELTY_BEARING":
        return "PRIMARY_POSITIVE_BASIS_CANDIDATE"
    if role_relevance == "REQUIRED_ENABLING":
        return "ENABLING_POSITIVE_BASIS_CANDIDATE"
    return "NONQUALIFYING_TENSION_ONLY"


def _future_readiness(
    *,
    qualifying_basis_present: bool,
    nonqualifying_tension_present: bool,
    direct_prior_art_blocker_present: bool,
    all_claims_fully_classified: bool,
) -> FutureAdjudicationReadiness:
    if qualifying_basis_present:
        if direct_prior_art_blocker_present:
            return (
                "BASIS_PRESENT_BUT_DIRECT_PRIOR_ART_BLOCKS_CERTIFICATION"
            )
        if not all_claims_fully_classified:
            return (
                "BASIS_PRESENT_BUT_EPISTEMICALLY_PARTIAL"
            )
        return (
            "READY_FOR_POSITIVE_NONOBVIOUSNESS_ADJUDICATION"
        )
    if nonqualifying_tension_present:
        return "NONQUALIFYING_TENSION_ONLY"
    return "NO_POSITIVE_BASIS"


def build_positive_nonobviousness_basis_report(
    *,
    graph: ScientificClaimEvidenceGraphReport,
    aggregation: ScientificHypothesisEvidenceAggregationReport,
) -> PositiveNonObviousnessBasisReport:
    if (
        aggregation.source_claim_evidence_graph_id
        != graph.graph_id
    ):
        raise ValueError(
            "aggregation report does not derive from supplied evidence graph"
        )

    graph_claims = {
        row.claim_id: row
        for row in graph.claim_summaries
    }
    impacts = {
        row.claim_id: row
        for row in aggregation.claim_impacts
    }
    if set(graph_claims) != set(impacts):
        raise ValueError(
            "aggregation/evidence graph claim sets differ"
        )

    work_id_by_node = _work_id_by_node(graph)

    basis_by_claim: dict[
        str,
        list[PositiveNonObviousnessBasisEvidence],
    ] = defaultdict(list)

    for edge in graph.edges:
        if (
            edge.edge_kind
            != "WORK_ADJUDICATED_TO_CLAIM"
            or edge.claim_id is None
            or edge.relationship
            not in _RELATIONSHIP_TO_BASIS_KIND
        ):
            continue

        if not edge.evidence_span.strip():
            raise ValueError(
                "positive basis source edge missing evidence span"
            )

        work_id = work_id_by_node.get(
            edge.source_node_id
        )
        if work_id is None:
            raise ValueError(
                "positive basis source is not a known work node"
            )

        basis_by_claim[edge.claim_id].append(
            PositiveNonObviousnessBasisEvidence(
                hypothesis_id=(
                    edge.hypothesis_id
                    or impacts[
                        edge.claim_id
                    ].hypothesis_id
                ),
                claim_id=edge.claim_id,
                work_id=work_id,
                relationship=edge.relationship,
                basis_kind=_RELATIONSHIP_TO_BASIS_KIND[
                    edge.relationship
                ],
                evidence_span=edge.evidence_span,
                rationale=edge.rationale,
                confidence=edge.confidence,
                basis_projection_ids=[],
                deterministic_reason_codes=list(
                    edge.deterministic_reason_codes
                ),
            )
        )

    claim_records: list[
        ClaimPositiveNonObviousnessBasisRecord
    ] = []

    for claim_id in sorted(graph_claims):
        impact = impacts[claim_id]
        evidence = sorted(
            basis_by_claim.get(claim_id, []),
            key=lambda row: (
                row.basis_kind,
                row.work_id,
            ),
        )
        role_relevance = _ROLE_RELEVANCE.get(
            impact.novelty_selection_role,
            "AUXILIARY_ONLY",
        )
        basis_state = _basis_state(
            role_relevance=role_relevance,
            evidence_count=len(evidence),
        )

        claim_records.append(
            ClaimPositiveNonObviousnessBasisRecord(
                hypothesis_id=impact.hypothesis_id,
                claim_id=claim_id,
                novelty_selection_role=(
                    impact.novelty_selection_role
                ),
                role_relevance=role_relevance,
                structural_centrality_score=(
                    impact.structural_centrality_score
                ),
                centrality_rank_within_hypothesis=(
                    impact.centrality_rank_within_hypothesis
                ),
                highest_centrality_within_hypothesis=(
                    impact.highest_centrality_within_hypothesis
                ),
                classification_fraction=(
                    impact.classification_fraction
                ),
                classification_coverage_state=(
                    impact.classification_coverage_state
                ),
                basis_evidence=evidence,
                basis_evidence_count=len(evidence),
                basis_kinds=sorted(
                    set(row.basis_kind for row in evidence)
                ),
                basis_work_ids=sorted(
                    set(row.work_id for row in evidence)
                ),
                direct_prior_art_work_ids=list(
                    impact.direct_prior_art_work_ids
                ),
                lower_order_prior_art_work_ids=list(
                    impact.lower_order_prior_art_work_ids
                ),
                direct_prior_art_blocker_present=bool(
                    impact.direct_prior_art_work_ids
                ),
                lower_order_prior_art_pressure_present=bool(
                    impact.lower_order_prior_art_work_ids
                ),
                basis_state=basis_state,
                qualifying_for_future_nonobviousness_adjudication=bool(
                    evidence
                    and role_relevance
                    in _QUALIFYING_ROLE_RELEVANCE
                ),
            )
        )

    by_hypothesis: dict[
        str,
        list[ClaimPositiveNonObviousnessBasisRecord],
    ] = defaultdict(list)
    for row in claim_records:
        by_hypothesis[row.hypothesis_id].append(row)

    agg_by_hypothesis = {
        row.hypothesis_id: row
        for row in aggregation.hypothesis_aggregations
    }

    summaries: list[
        HypothesisPositiveNonObviousnessBasisSummary
    ] = []

    for hypothesis_id in sorted(by_hypothesis):
        rows = by_hypothesis[hypothesis_id]
        agg = agg_by_hypothesis[hypothesis_id]

        qualifying = {
            row.claim_id
            for row in rows
            if row.qualifying_for_future_nonobviousness_adjudication
        }
        nonqualifying = {
            row.claim_id
            for row in rows
            if (
                row.basis_evidence_count > 0
                and not row.qualifying_for_future_nonobviousness_adjudication
            )
        }
        no_basis = {
            row.claim_id
            for row in rows
            if row.basis_evidence_count == 0
        }
        direct_blockers = {
            row.claim_id
            for row in rows
            if row.direct_prior_art_blocker_present
        }
        lower_pressure = {
            row.claim_id
            for row in rows
            if row.lower_order_prior_art_pressure_present
        }

        if qualifying:
            positive_state = (
                "QUALIFYING_POSITIVE_BASIS_CANDIDATE_PRESENT"
            )
        elif nonqualifying:
            positive_state = "NONQUALIFYING_TENSION_ONLY"
        else:
            positive_state = "NO_POSITIVE_BASIS_CANDIDATE"

        basis_work_ids = sorted(
            {
                evidence.work_id
                for row in rows
                for evidence in row.basis_evidence
            }
        )
        kind_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            for evidence in row.basis_evidence:
                kind_counts[evidence.basis_kind] += 1

        summaries.append(
            HypothesisPositiveNonObviousnessBasisSummary(
                hypothesis_id=hypothesis_id,
                claim_ids=sorted(
                    row.claim_id
                    for row in rows
                ),
                qualifying_basis_claim_ids=sorted(
                    qualifying
                ),
                nonqualifying_tension_claim_ids=sorted(
                    nonqualifying
                ),
                no_basis_claim_ids=sorted(no_basis),
                basis_work_ids=basis_work_ids,
                basis_kind_counts=dict(
                    sorted(kind_counts.items())
                ),
                direct_prior_art_blocker_claim_ids=sorted(
                    direct_blockers
                ),
                lower_order_pressure_claim_ids=sorted(
                    lower_pressure
                ),
                positive_basis_state=positive_state,
                future_adjudication_readiness=_future_readiness(
                    qualifying_basis_present=bool(
                        qualifying
                    ),
                    nonqualifying_tension_present=bool(
                        nonqualifying
                    ),
                    direct_prior_art_blocker_present=bool(
                        direct_blockers
                    ),
                    all_claims_fully_classified=(
                        agg.all_claims_fully_classified
                    ),
                ),
                epistemic_coverage_profile=(
                    agg.epistemic_coverage_profile
                ),
                all_claims_fully_classified=(
                    agg.all_claims_fully_classified
                ),
                any_unclassified_presented_work=(
                    agg.any_unclassified_presented_work
                ),
                direct_prior_art_blocks_future_certification=bool(
                    direct_blockers
                ),
            )
        )

    state_counts: dict[str, int] = defaultdict(int)
    readiness_counts: dict[str, int] = defaultdict(int)
    for row in summaries:
        state_counts[row.positive_basis_state] += 1
        readiness_counts[
            row.future_adjudication_readiness
        ] += 1

    body = {
        "schema_version":
            "positive-nonobviousness-basis-report-v1",
        "source_claim_evidence_graph_id":
            graph.graph_id,
        "source_hypothesis_evidence_aggregation_report_id":
            aggregation.report_id,
        "claim_records": [
            row.model_dump(mode="json")
            for row in claim_records
        ],
        "hypothesis_summaries": [
            row.model_dump(mode="json")
            for row in summaries
        ],
        "claim_count":
            len(claim_records),
        "hypothesis_count":
            len(summaries),
        "basis_evidence_count":
            sum(
                row.basis_evidence_count
                for row in claim_records
            ),
        "qualifying_basis_claim_count":
            sum(
                row.qualifying_for_future_nonobviousness_adjudication
                for row in claim_records
            ),
        "positive_basis_state_counts":
            dict(sorted(state_counts.items())),
        "readiness_counts":
            dict(sorted(readiness_counts.items())),
        "basis_semantics":
            "adjudicated_tension_conflict_only_no_absence_inference_v1",
        "direct_prior_art_is_not_positive_basis":
            True,
        "lower_order_prior_art_is_not_positive_basis":
            True,
        "partial_prior_art_is_not_positive_basis":
            True,
        "component_only_is_not_positive_basis":
            True,
        "retrieval_miss_is_not_positive_basis":
            True,
        "positive_basis_is_not_truth_evidence":
            True,
        "positive_basis_is_not_novelty_verdict":
            True,
        "positive_basis_is_not_nonobviousness_verdict":
            True,
        "positive_nonobviousness_authority_created":
            False,
        "novelty_verdict_created":
            False,
        "certification_performed":
            False,
        "diagnostic_only":
            True,
        "n9_contract_changed":
            False,
        "n10_contract_changed":
            False,
        "production_selection_changed":
            False,
    }
    digest = _sha256_json(body)

    return PositiveNonObviousnessBasisReport(
        **body,
        report_id=(
            "positive_nonobviousness_basis:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "ClaimPositiveNonObviousnessBasisRecord",
    "HypothesisPositiveNonObviousnessBasisSummary",
    "PositiveNonObviousnessBasisEvidence",
    "PositiveNonObviousnessBasisReport",
    "build_positive_nonobviousness_basis_report",
]
