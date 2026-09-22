from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.scientific_claim_centrality import (
    ClaimEvidencePressureState,
    ScientificClaimCentralityReport,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


StrongPressureKind = Literal[
    "DIRECT_PRIOR_ART",
    "LOWER_ORDER_RELATION_PRIOR_ART",
    "DIRECTIONAL_COUNTEREVIDENCE",
    "CONTEXTUAL_CONFLICT",
    "CONFLICTING_PRIOR_ART",
]

WeakPressureKind = Literal[
    "PARTIAL_PRIOR_ART",
    "COMPONENT_ONLY",
]

HypothesisEvidenceImpactProfile = Literal[
    "NOVELTY_BEARING_HIGHEST_CENTRALITY_STRONG_PRESSURE",
    "NOVELTY_BEARING_STRONG_PRESSURE_OUTSIDE_HIGHEST_CENTRALITY",
    "HIGHEST_CENTRALITY_STRONG_PRESSURE_OUTSIDE_NOVELTY_BEARING",
    "OTHER_STRONG_PRESSURE_ONLY",
    "NO_STRONG_RELATION_PRESSURE",
]

HypothesisWeakSignalProfile = Literal[
    "PARTIAL_PRIOR_ART_PRESENT",
    "COMPONENT_CONTEXT_ONLY",
    "NO_WEAK_OR_COMPONENT_SIGNAL",
]

EpistemicCoverageProfile = Literal[
    "FULL_CLASSIFICATION_COVERAGE",
    "PARTIAL_CLASSIFICATION_COVERAGE",
]

_STRONG_RELATIONSHIPS = frozenset(
    {
        "DIRECT_PRIOR_ART",
        "LOWER_ORDER_RELATION_PRIOR_ART",
        "DIRECTIONAL_COUNTEREVIDENCE",
        "CONTEXTUAL_CONFLICT",
        "CONFLICTING_PRIOR_ART",
    }
)

_WEAK_RELATIONSHIPS = frozenset(
    {
        "PARTIAL_PRIOR_ART",
        "COMPONENT_ONLY",
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


class ScientificClaimEvidenceImpactRecord(StrictModel):
    hypothesis_id: str
    claim_id: str

    novelty_selection_role: str
    role_precedence: int = Field(ge=0)

    topology_state: str
    structural_centrality_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    centrality_rank_within_hypothesis: int = Field(ge=1)
    highest_centrality_within_hypothesis: bool
    centrality_rank_tied: bool

    classification_coverage_state: str
    classification_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    evidence_pressure_state: ClaimEvidencePressureState

    strong_pressure_kinds: list[StrongPressureKind]
    strong_pressure_work_ids: list[str]
    weak_pressure_kinds: list[WeakPressureKind]
    weak_pressure_work_ids: list[str]

    direct_prior_art_work_ids: list[str]
    lower_order_prior_art_work_ids: list[str]
    directional_counterevidence_work_ids: list[str]
    contextual_conflict_work_ids: list[str]
    conflicting_prior_art_work_ids: list[str]
    partial_prior_art_work_ids: list[str]
    component_only_work_ids: list[str]

    strong_pressure_present: bool
    weak_or_component_signal_present: bool

    # This tuple is exposed for deterministic ordering/audit only. It is not a
    # scalar novelty or non-obviousness score.
    aggregation_order_key: list[float]

    evidence_pressure_is_bounded_to_classified_subset: Literal[
        True
    ] = True
    unclassified_is_not_negative_evidence: Literal[True] = True
    absence_based_novelty_authorized: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_pressure(self) -> "ScientificClaimEvidenceImpactRecord":
        expected_strong = bool(self.strong_pressure_work_ids)
        if self.strong_pressure_present != expected_strong:
            raise ValueError(
                "strong pressure presence mismatch"
            )
        expected_weak = bool(self.weak_pressure_work_ids)
        if self.weak_or_component_signal_present != expected_weak:
            raise ValueError(
                "weak pressure presence mismatch"
            )
        return self


class ScientificHypothesisEvidenceAggregation(StrictModel):
    hypothesis_id: str
    topology_state: str
    centrality_is_degenerate: bool

    claim_ids: list[str]
    novelty_bearing_claim_ids: list[str]
    highest_centrality_claim_ids: list[str]

    strong_pressure_claim_ids: list[str]
    novelty_bearing_strong_pressure_claim_ids: list[str]
    highest_centrality_strong_pressure_claim_ids: list[str]
    novelty_bearing_highest_centrality_strong_pressure_claim_ids: list[str]

    weak_or_component_signal_claim_ids: list[str]
    no_strong_pressure_claim_ids: list[str]

    strong_pressure_work_ids: list[str]
    weak_pressure_work_ids: list[str]

    strong_pressure_kind_counts: dict[str, int]
    weak_pressure_kind_counts: dict[str, int]

    evidence_impact_profile: HypothesisEvidenceImpactProfile
    weak_signal_profile: HypothesisWeakSignalProfile
    epistemic_coverage_profile: EpistemicCoverageProfile

    all_claims_fully_classified: bool
    any_unclassified_presented_work: bool

    # Ordered first by role precedence, then structural centrality rank, then
    # evidence pressure presence. This is a review queue, not a verdict.
    review_priority_claim_ids: list[str]

    # Strong prior-art pressure may challenge a central/novelty-bearing claim,
    # but this aggregation layer still does not infer novelty/non-obviousness.
    aggregation_semantics: Literal[
        "role_then_structural_centrality_then_bounded_evidence_pressure_v1"
    ] = "role_then_structural_centrality_then_bounded_evidence_pressure_v1"

    centrality_threshold_invented: Literal[False] = False
    weighted_novelty_score_computed: Literal[False] = False
    unclassified_is_not_negative_evidence: Literal[True] = True
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_sets(
        self,
    ) -> "ScientificHypothesisEvidenceAggregation":
        claims = set(self.claim_ids)
        for label, values in (
            (
                "novelty_bearing_claim_ids",
                self.novelty_bearing_claim_ids,
            ),
            (
                "highest_centrality_claim_ids",
                self.highest_centrality_claim_ids,
            ),
            (
                "strong_pressure_claim_ids",
                self.strong_pressure_claim_ids,
            ),
            (
                "novelty_bearing_strong_pressure_claim_ids",
                self.novelty_bearing_strong_pressure_claim_ids,
            ),
            (
                "highest_centrality_strong_pressure_claim_ids",
                self.highest_centrality_strong_pressure_claim_ids,
            ),
            (
                "novelty_bearing_highest_centrality_strong_pressure_claim_ids",
                self.novelty_bearing_highest_centrality_strong_pressure_claim_ids,
            ),
            (
                "weak_or_component_signal_claim_ids",
                self.weak_or_component_signal_claim_ids,
            ),
            (
                "no_strong_pressure_claim_ids",
                self.no_strong_pressure_claim_ids,
            ),
            (
                "review_priority_claim_ids",
                self.review_priority_claim_ids,
            ),
        ):
            if not set(values).issubset(claims):
                raise ValueError(
                    f"{label} contains unknown claim IDs"
                )
        if set(self.review_priority_claim_ids) != claims:
            raise ValueError(
                "review priority must cover every claim exactly once"
            )
        if len(self.review_priority_claim_ids) != len(claims):
            raise ValueError(
                "review priority contains duplicate claim IDs"
            )
        return self


class ScientificHypothesisEvidenceAggregationReport(StrictModel):
    schema_version: Literal[
        "scientific-hypothesis-evidence-aggregation-report-v1"
    ] = "scientific-hypothesis-evidence-aggregation-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_claim_evidence_graph_id: str
    source_claim_centrality_report_id: str

    claim_impacts: list[ScientificClaimEvidenceImpactRecord]
    hypothesis_aggregations: list[
        ScientificHypothesisEvidenceAggregation
    ]

    claim_count: int = Field(ge=0)
    hypothesis_count: int = Field(ge=0)

    hypothesis_profile_counts: dict[str, int]
    epistemic_coverage_profile_counts: dict[str, int]

    aggregation_semantics: Literal[
        "centrality_aware_evidence_impact_without_novelty_verdict_v1"
    ] = "centrality_aware_evidence_impact_without_novelty_verdict_v1"

    role_precedence_is_lexicographic_not_numeric_weight: Literal[
        True
    ] = True
    centrality_used_as_relative_topology_not_novelty_score: Literal[
        True
    ] = True
    evidence_pressure_bounded_to_classified_subset: Literal[True] = True
    unclassified_is_not_negative_evidence: Literal[True] = True
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    novelty_verdict_created: Literal[False] = False
    diagnostic_only: Literal[True] = True
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ScientificHypothesisEvidenceAggregationReport":
        if self.claim_count != len(self.claim_impacts):
            raise ValueError(
                "aggregation claim_count mismatch"
            )
        if self.hypothesis_count != len(
            self.hypothesis_aggregations
        ):
            raise ValueError(
                "aggregation hypothesis_count mismatch"
            )

        expected_profiles: dict[str, int] = defaultdict(int)
        expected_coverage: dict[str, int] = defaultdict(int)
        for row in self.hypothesis_aggregations:
            expected_profiles[row.evidence_impact_profile] += 1
            expected_coverage[row.epistemic_coverage_profile] += 1

        if dict(sorted(expected_profiles.items())) != dict(
            sorted(self.hypothesis_profile_counts.items())
        ):
            raise ValueError(
                "hypothesis profile counts mismatch"
            )
        if dict(sorted(expected_coverage.items())) != dict(
            sorted(
                self.epistemic_coverage_profile_counts.items()
            )
        ):
            raise ValueError(
                "epistemic coverage profile counts mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "hypothesis evidence aggregation SHA mismatch"
            )
        if observed_id != (
            "scientific_hypothesis_evidence_aggregation:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "hypothesis evidence aggregation ID mismatch"
            )
        return self


def _relationship_work_ids(
    *,
    graph: ScientificClaimEvidenceGraphReport,
    claim_id: str,
) -> dict[str, list[str]]:
    rows: dict[str, set[str]] = defaultdict(set)

    for edge in graph.edges:
        if (
            edge.edge_kind
            != "WORK_ADJUDICATED_TO_CLAIM"
            or edge.claim_id != claim_id
            or edge.relationship is None
        ):
            continue

        work_node = next(
            (
                node
                for node in graph.nodes
                if node.node_id == edge.source_node_id
                and node.node_kind == "WORK"
            ),
            None,
        )
        if work_node is None:
            raise ValueError(
                "adjudication edge source is not a WORK node"
            )

        if work_node.source_ids:
            work_id = work_node.source_ids[0]
        else:
            work_id = work_node.node_id

        rows[edge.relationship].add(work_id)

    return {
        relationship: sorted(work_ids)
        for relationship, work_ids in rows.items()
    }


def _impact_profile(
    *,
    novelty_bearing_strong: set[str],
    highest_centrality_strong: set[str],
    novelty_bearing_highest_strong: set[str],
    strong: set[str],
) -> HypothesisEvidenceImpactProfile:
    if novelty_bearing_highest_strong:
        return (
            "NOVELTY_BEARING_HIGHEST_CENTRALITY_STRONG_PRESSURE"
        )
    if novelty_bearing_strong:
        return (
            "NOVELTY_BEARING_STRONG_PRESSURE_OUTSIDE_HIGHEST_CENTRALITY"
        )
    if highest_centrality_strong:
        return (
            "HIGHEST_CENTRALITY_STRONG_PRESSURE_OUTSIDE_NOVELTY_BEARING"
        )
    if strong:
        return "OTHER_STRONG_PRESSURE_ONLY"
    return "NO_STRONG_RELATION_PRESSURE"


def _weak_signal_profile(
    claim_rows: list[ScientificClaimEvidenceImpactRecord],
) -> HypothesisWeakSignalProfile:
    if any(
        "PARTIAL_PRIOR_ART" in row.weak_pressure_kinds
        for row in claim_rows
    ):
        return "PARTIAL_PRIOR_ART_PRESENT"
    if any(
        "COMPONENT_ONLY" in row.weak_pressure_kinds
        for row in claim_rows
    ):
        return "COMPONENT_CONTEXT_ONLY"
    return "NO_WEAK_OR_COMPONENT_SIGNAL"


def build_scientific_hypothesis_evidence_aggregation(
    *,
    graph: ScientificClaimEvidenceGraphReport,
    centrality: ScientificClaimCentralityReport,
) -> ScientificHypothesisEvidenceAggregationReport:
    if (
        centrality.source_claim_evidence_graph_id
        != graph.graph_id
    ):
        raise ValueError(
            "centrality report does not derive from supplied evidence graph"
        )

    graph_claim_ids = {
        row.claim_id
        for row in graph.claim_summaries
    }
    centrality_claim_ids = {
        row.claim_id
        for row in centrality.claim_records
    }
    if graph_claim_ids != centrality_claim_ids:
        raise ValueError(
            "centrality/evidence graph claim sets differ"
        )

    centrality_by_claim = {
        row.claim_id: row
        for row in centrality.claim_records
    }

    claim_impacts: list[ScientificClaimEvidenceImpactRecord] = []

    for summary in graph.claim_summaries:
        row = centrality_by_claim[summary.claim_id]
        relationship_works = _relationship_work_ids(
            graph=graph,
            claim_id=summary.claim_id,
        )

        direct = relationship_works.get(
            "DIRECT_PRIOR_ART",
            [],
        )
        lower = relationship_works.get(
            "LOWER_ORDER_RELATION_PRIOR_ART",
            [],
        )
        directional = relationship_works.get(
            "DIRECTIONAL_COUNTEREVIDENCE",
            [],
        )
        contextual = relationship_works.get(
            "CONTEXTUAL_CONFLICT",
            [],
        )
        conflicting = relationship_works.get(
            "CONFLICTING_PRIOR_ART",
            [],
        )
        partial = relationship_works.get(
            "PARTIAL_PRIOR_ART",
            [],
        )
        component = relationship_works.get(
            "COMPONENT_ONLY",
            [],
        )

        strong_kinds: list[StrongPressureKind] = []
        for kind, works in (
            ("DIRECT_PRIOR_ART", direct),
            ("LOWER_ORDER_RELATION_PRIOR_ART", lower),
            ("DIRECTIONAL_COUNTEREVIDENCE", directional),
            ("CONTEXTUAL_CONFLICT", contextual),
            ("CONFLICTING_PRIOR_ART", conflicting),
        ):
            if works:
                strong_kinds.append(kind)

        weak_kinds: list[WeakPressureKind] = []
        for kind, works in (
            ("PARTIAL_PRIOR_ART", partial),
            ("COMPONENT_ONLY", component),
        ):
            if works:
                weak_kinds.append(kind)

        strong_work_ids = sorted(
            set(
                direct
                + lower
                + directional
                + contextual
                + conflicting
            )
        )
        weak_work_ids = sorted(
            set(partial + component)
        )

        # Review ordering is lexicographic and auditable:
        #   role precedence -> centrality rank -> strong pressure presence ->
        #   weak/context signal presence -> claim ID.
        # No weighted novelty score is computed.
        aggregation_order_key = [
            float(row.role_precedence),
            float(row.centrality_rank_within_hypothesis),
            0.0 if strong_work_ids else 1.0,
            0.0 if weak_work_ids else 1.0,
        ]

        claim_impacts.append(
            ScientificClaimEvidenceImpactRecord(
                hypothesis_id=row.hypothesis_id,
                claim_id=row.claim_id,
                novelty_selection_role=(
                    row.novelty_selection_role
                ),
                role_precedence=row.role_precedence,
                topology_state=row.topology_state,
                structural_centrality_score=(
                    row.structural_centrality_score
                ),
                centrality_rank_within_hypothesis=(
                    row.centrality_rank_within_hypothesis
                ),
                highest_centrality_within_hypothesis=(
                    row.centrality_rank_within_hypothesis == 1
                ),
                centrality_rank_tied=row.centrality_rank_tied,
                classification_coverage_state=(
                    row.classification_coverage_state
                ),
                classification_fraction=(
                    row.classification_fraction
                ),
                evidence_pressure_state=(
                    row.evidence_pressure_state
                ),
                strong_pressure_kinds=strong_kinds,
                strong_pressure_work_ids=strong_work_ids,
                weak_pressure_kinds=weak_kinds,
                weak_pressure_work_ids=weak_work_ids,
                direct_prior_art_work_ids=direct,
                lower_order_prior_art_work_ids=lower,
                directional_counterevidence_work_ids=directional,
                contextual_conflict_work_ids=contextual,
                conflicting_prior_art_work_ids=conflicting,
                partial_prior_art_work_ids=partial,
                component_only_work_ids=component,
                strong_pressure_present=bool(
                    strong_work_ids
                ),
                weak_or_component_signal_present=bool(
                    weak_work_ids
                ),
                aggregation_order_key=aggregation_order_key,
            )
        )

    claim_impacts.sort(
        key=lambda row: (
            row.hypothesis_id,
            *row.aggregation_order_key,
            row.claim_id,
        )
    )

    by_hypothesis: dict[
        str,
        list[ScientificClaimEvidenceImpactRecord],
    ] = defaultdict(list)
    for row in claim_impacts:
        by_hypothesis[row.hypothesis_id].append(row)

    centrality_summary_by_hypothesis = {
        row.hypothesis_id: row
        for row in centrality.hypothesis_summaries
    }

    hypothesis_aggregations: list[
        ScientificHypothesisEvidenceAggregation
    ] = []

    for hypothesis_id in sorted(by_hypothesis):
        rows = by_hypothesis[hypothesis_id]
        centrality_summary = (
            centrality_summary_by_hypothesis[hypothesis_id]
        )

        claim_ids = {
            row.claim_id
            for row in rows
        }
        novelty_bearing = {
            row.claim_id
            for row in rows
            if row.novelty_selection_role
            == "NOVELTY_BEARING"
        }
        highest = {
            row.claim_id
            for row in rows
            if row.highest_centrality_within_hypothesis
        }
        strong = {
            row.claim_id
            for row in rows
            if row.strong_pressure_present
        }
        novelty_bearing_strong = (
            novelty_bearing & strong
        )
        highest_strong = highest & strong
        novelty_bearing_highest_strong = (
            novelty_bearing
            & highest
            & strong
        )
        weak = {
            row.claim_id
            for row in rows
            if row.weak_or_component_signal_present
        }

        strong_kind_counts: dict[str, int] = defaultdict(int)
        weak_kind_counts: dict[str, int] = defaultdict(int)
        strong_work_ids: set[str] = set()
        weak_work_ids: set[str] = set()

        for row in rows:
            for kind in row.strong_pressure_kinds:
                strong_kind_counts[kind] += 1
            for kind in row.weak_pressure_kinds:
                weak_kind_counts[kind] += 1
            strong_work_ids.update(
                row.strong_pressure_work_ids
            )
            weak_work_ids.update(
                row.weak_pressure_work_ids
            )

        any_unclassified = any(
            row.classification_coverage_state
            in {
                "PARTIALLY_CLASSIFIED",
                "UNCLASSIFIED_ONLY",
            }
            for row in rows
        )
        all_fully_classified = all(
            row.classification_coverage_state
            in {
                "FULLY_CLASSIFIED",
                "NO_PRESENTED_WORKS",
            }
            for row in rows
        )

        profile = _impact_profile(
            novelty_bearing_strong=novelty_bearing_strong,
            highest_centrality_strong=highest_strong,
            novelty_bearing_highest_strong=(
                novelty_bearing_highest_strong
            ),
            strong=strong,
        )

        review_priority = [
            row.claim_id
            for row in sorted(
                rows,
                key=lambda row: (
                    *row.aggregation_order_key,
                    row.claim_id,
                ),
            )
        ]

        hypothesis_aggregations.append(
            ScientificHypothesisEvidenceAggregation(
                hypothesis_id=hypothesis_id,
                topology_state=(
                    centrality_summary.topology_state
                ),
                centrality_is_degenerate=(
                    centrality_summary.centrality_is_degenerate
                ),
                claim_ids=sorted(claim_ids),
                novelty_bearing_claim_ids=sorted(
                    novelty_bearing
                ),
                highest_centrality_claim_ids=sorted(
                    highest
                ),
                strong_pressure_claim_ids=sorted(strong),
                novelty_bearing_strong_pressure_claim_ids=sorted(
                    novelty_bearing_strong
                ),
                highest_centrality_strong_pressure_claim_ids=sorted(
                    highest_strong
                ),
                novelty_bearing_highest_centrality_strong_pressure_claim_ids=sorted(
                    novelty_bearing_highest_strong
                ),
                weak_or_component_signal_claim_ids=sorted(
                    weak
                ),
                no_strong_pressure_claim_ids=sorted(
                    claim_ids - strong
                ),
                strong_pressure_work_ids=sorted(
                    strong_work_ids
                ),
                weak_pressure_work_ids=sorted(
                    weak_work_ids
                ),
                strong_pressure_kind_counts=dict(
                    sorted(strong_kind_counts.items())
                ),
                weak_pressure_kind_counts=dict(
                    sorted(weak_kind_counts.items())
                ),
                evidence_impact_profile=profile,
                weak_signal_profile=_weak_signal_profile(
                    rows
                ),
                epistemic_coverage_profile=(
                    "FULL_CLASSIFICATION_COVERAGE"
                    if all_fully_classified
                    else "PARTIAL_CLASSIFICATION_COVERAGE"
                ),
                all_claims_fully_classified=(
                    all_fully_classified
                ),
                any_unclassified_presented_work=(
                    any_unclassified
                ),
                review_priority_claim_ids=(
                    review_priority
                ),
            )
        )

    profile_counts: dict[str, int] = defaultdict(int)
    coverage_counts: dict[str, int] = defaultdict(int)
    for row in hypothesis_aggregations:
        profile_counts[row.evidence_impact_profile] += 1
        coverage_counts[row.epistemic_coverage_profile] += 1

    body = {
        "schema_version":
            "scientific-hypothesis-evidence-aggregation-report-v1",
        "source_claim_evidence_graph_id":
            graph.graph_id,
        "source_claim_centrality_report_id":
            centrality.report_id,
        "claim_impacts": [
            row.model_dump(mode="json")
            for row in claim_impacts
        ],
        "hypothesis_aggregations": [
            row.model_dump(mode="json")
            for row in hypothesis_aggregations
        ],
        "claim_count":
            len(claim_impacts),
        "hypothesis_count":
            len(hypothesis_aggregations),
        "hypothesis_profile_counts":
            dict(sorted(profile_counts.items())),
        "epistemic_coverage_profile_counts":
            dict(sorted(coverage_counts.items())),
        "aggregation_semantics":
            "centrality_aware_evidence_impact_without_novelty_verdict_v1",
        "role_precedence_is_lexicographic_not_numeric_weight":
            True,
        "centrality_used_as_relative_topology_not_novelty_score":
            True,
        "evidence_pressure_bounded_to_classified_subset":
            True,
        "unclassified_is_not_negative_evidence":
            True,
        "absence_based_novelty_authorized":
            False,
        "positive_nonobviousness_authority_created":
            False,
        "novelty_verdict_created":
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

    return ScientificHypothesisEvidenceAggregationReport(
        **body,
        report_id=(
            "scientific_hypothesis_evidence_aggregation:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "ScientificClaimEvidenceImpactRecord",
    "ScientificHypothesisEvidenceAggregation",
    "ScientificHypothesisEvidenceAggregationReport",
    "build_scientific_hypothesis_evidence_aggregation",
]
