from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
    GroundedFactorRelationProjection,
)
from pipeline_core.discovery.projection_relation_adjudication import (
    ProjectionPriorArtRelationship,
    ProjectionRelationAdjudicationReport,
    ProjectionRelationCandidateReport,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificConceptIR,
    ScientificRelationIR,
    ScientificRelationIRReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


GraphNodeKind = Literal[
    "HYPOTHESIS",
    "CLAIM",
    "CONCEPT",
    "PROJECTION",
    "WORK",
]

GraphEdgeKind = Literal[
    "HYPOTHESIS_HAS_CLAIM",
    "CLAIM_HAS_CONCEPT",
    "CLAIM_HAS_PROJECTION",
    "CLAIM_PRESENTED_WORK",
    "WORK_ADJUDICATED_TO_CLAIM",
    "WORK_BASIS_PROJECTION",
]

ClaimAdjudicationStatus = Literal[
    "REVIEWED",
    "NOT_REVIEWED",
]

_STRONG_RELATIONSHIPS = frozenset(
    {
        "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART",
        "LOWER_ORDER_RELATION_PRIOR_ART",
        "DIRECTIONAL_COUNTEREVIDENCE",
        "CONTEXTUAL_CONFLICT",
        "CONFLICTING_PRIOR_ART",
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


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


class ScientificClaimEvidenceGraphNode(StrictModel):
    node_id: str
    node_kind: GraphNodeKind
    label: str

    hypothesis_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    # CLAIM metadata.
    relation_ir_id: str | None = None
    claim_kind: str | None = None
    novelty_selection_role: str | None = None
    typing_status: str | None = None
    relation_state: str | None = None

    # CONCEPT metadata. Exact normalized text + typed identity is the only
    # deterministic cross-claim concept coalescing performed here.
    normalized_text: str | None = None
    concept_roles: list[str] = Field(default_factory=list)
    type_labels: list[str] = Field(default_factory=list)

    # PROJECTION metadata.
    projection_kind: str | None = None
    factor_order: int | None = Field(default=None, ge=0)
    retained_factor_ids: list[str] = Field(default_factory=list)
    omitted_factor_ids: list[str] = Field(default_factory=list)

    # WORK metadata.
    title: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    source_work_ids: list[str] = Field(default_factory=list)

    novelty_authority: Literal[False] = False


class ScientificClaimEvidenceGraphEdge(StrictModel):
    edge_id: str
    edge_kind: GraphEdgeKind
    source_node_id: str
    target_node_id: str

    hypothesis_id: str | None = None
    claim_id: str | None = None

    concept_role: str | None = None

    review_status: Literal[
        "CLASSIFIED",
        "UNCLASSIFIED",
    ] | None = None
    typed_compatibility_state: str | None = None
    selected_lanes: list[str] = Field(default_factory=list)
    relation_anchor_tier: str | None = None
    endpoint_supported_count: int | None = Field(default=None, ge=0)
    second_pass_roles: list[str] = Field(default_factory=list)
    counterevidence_modes: list[str] = Field(default_factory=list)

    relationship: ProjectionPriorArtRelationship | None = None
    original_relationship: ProjectionPriorArtRelationship | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_span: str = ""
    rationale: str = ""
    deterministic_reason_codes: list[str] = Field(default_factory=list)

    basis_projection_id: str | None = None

    retrieval_is_not_relation_evidence: Literal[True] = True
    unclassified_is_not_negative_evidence: Literal[True] = True
    absence_based_novelty_authorized: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_edge_semantics(self) -> "ScientificClaimEvidenceGraphEdge":
        if self.edge_kind == "CLAIM_PRESENTED_WORK":
            if self.review_status is None:
                raise ValueError(
                    "presentation edge requires review_status"
                )
            if self.relationship is not None:
                raise ValueError(
                    "presentation edge cannot carry adjudicated relationship"
                )

        if self.edge_kind == "WORK_ADJUDICATED_TO_CLAIM":
            if self.review_status != "CLASSIFIED":
                raise ValueError(
                    "adjudication edge must be CLASSIFIED"
                )
            if self.relationship is None:
                raise ValueError(
                    "adjudication edge requires relationship"
                )
            if (
                self.relationship in _STRONG_RELATIONSHIPS
                and not self.evidence_span.strip()
            ):
                raise ValueError(
                    "strong adjudicated relationship requires evidence span"
                )

        if self.edge_kind == "WORK_BASIS_PROJECTION":
            if not self.basis_projection_id:
                raise ValueError(
                    "work-basis edge requires basis_projection_id"
                )

        return self


class ScientificClaimEvidenceGraphClaimSummary(StrictModel):
    hypothesis_id: str
    claim_id: str
    relation_ir_id: str
    claim_kind: str
    novelty_selection_role: str
    typing_status: str

    adjudication_status: ClaimAdjudicationStatus
    relation_state: str | None = None

    concept_node_ids: list[str]
    projection_node_ids: list[str]
    presented_work_node_ids: list[str]
    classified_work_node_ids: list[str]
    unclassified_work_node_ids: list[str]

    presented_work_count: int = Field(ge=0)
    classified_work_count: int = Field(ge=0)
    unclassified_work_count: int = Field(ge=0)

    relationship_counts: dict[str, int] = Field(default_factory=dict)

    no_absence_inference: Literal[True] = True
    unclassified_is_not_negative_evidence: Literal[True] = True
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "ScientificClaimEvidenceGraphClaimSummary":
        if self.presented_work_count != len(
            self.presented_work_node_ids
        ):
            raise ValueError("presented work count mismatch")
        if self.classified_work_count != len(
            self.classified_work_node_ids
        ):
            raise ValueError("classified work count mismatch")
        if self.unclassified_work_count != len(
            self.unclassified_work_node_ids
        ):
            raise ValueError("unclassified work count mismatch")
        if (
            self.presented_work_count
            != self.classified_work_count
            + self.unclassified_work_count
        ):
            raise ValueError(
                "claim evidence graph coverage mismatch"
            )
        return self


class ScientificClaimEvidenceGraphHypothesisSummary(StrictModel):
    hypothesis_id: str
    claim_ids: list[str]
    novelty_bearing_claim_ids: list[str]
    reviewed_claim_ids: list[str]
    unreviewed_claim_ids: list[str]

    claim_count: int = Field(ge=0)
    reviewed_claim_count: int = Field(ge=0)
    unreviewed_claim_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "ScientificClaimEvidenceGraphHypothesisSummary":
        if self.claim_count != len(self.claim_ids):
            raise ValueError("hypothesis claim_count mismatch")
        if self.reviewed_claim_count != len(
            self.reviewed_claim_ids
        ):
            raise ValueError(
                "hypothesis reviewed_claim_count mismatch"
            )
        if self.unreviewed_claim_count != len(
            self.unreviewed_claim_ids
        ):
            raise ValueError(
                "hypothesis unreviewed_claim_count mismatch"
            )
        if (
            self.claim_count
            != self.reviewed_claim_count
            + self.unreviewed_claim_count
        ):
            raise ValueError(
                "hypothesis review coverage mismatch"
            )
        return self


class ScientificClaimEvidenceGraphReport(StrictModel):
    schema_version: Literal[
        "scientific-claim-evidence-graph-report-v1"
    ] = "scientific-claim-evidence-graph-report-v1"

    graph_id: str
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_relation_ir_report_id: str
    source_projection_report_id: str
    source_candidate_report_id: str
    source_adjudication_report_id: str

    nodes: list[ScientificClaimEvidenceGraphNode]
    edges: list[ScientificClaimEvidenceGraphEdge]
    claim_summaries: list[
        ScientificClaimEvidenceGraphClaimSummary
    ]
    hypothesis_summaries: list[
        ScientificClaimEvidenceGraphHypothesisSummary
    ]

    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    hypothesis_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    concept_node_count: int = Field(ge=0)
    projection_node_count: int = Field(ge=0)
    work_node_count: int = Field(ge=0)
    presented_work_edge_count: int = Field(ge=0)
    adjudicated_relation_edge_count: int = Field(ge=0)
    unclassified_presentation_edge_count: int = Field(ge=0)

    graph_semantics: Literal[
        "deterministic_claim_evidence_topology_v1"
    ] = "deterministic_claim_evidence_topology_v1"

    exact_normalized_concept_coalescing_only: Literal[True] = True
    classified_and_unclassified_review_separated: Literal[True] = True
    unclassified_is_not_negative_evidence: Literal[True] = True
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    centrality_scoring_performed: Literal[False] = False
    aggregation_performed: Literal[False] = False
    diagnostic_only: Literal[True] = True
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_graph(self) -> "ScientificClaimEvidenceGraphReport":
        if self.node_count != len(self.nodes):
            raise ValueError("node_count mismatch")
        if self.edge_count != len(self.edges):
            raise ValueError("edge_count mismatch")
        if self.claim_count != len(self.claim_summaries):
            raise ValueError("claim_count mismatch")
        if self.hypothesis_count != len(
            self.hypothesis_summaries
        ):
            raise ValueError("hypothesis_count mismatch")

        node_ids = [row.node_id for row in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate evidence graph node ID")
        edge_ids = [row.edge_id for row in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("duplicate evidence graph edge ID")

        allowed = set(node_ids)
        for edge in self.edges:
            if edge.source_node_id not in allowed:
                raise ValueError(
                    "edge source missing from graph: "
                    + edge.source_node_id
                )
            if edge.target_node_id not in allowed:
                raise ValueError(
                    "edge target missing from graph: "
                    + edge.target_node_id
                )

        expected_kind_counts = {
            "CONCEPT": self.concept_node_count,
            "PROJECTION": self.projection_node_count,
            "WORK": self.work_node_count,
        }
        for kind, expected in expected_kind_counts.items():
            actual = sum(
                row.node_kind == kind
                for row in self.nodes
            )
            if actual != expected:
                raise ValueError(
                    f"{kind} node count mismatch"
                )

        presented = [
            row
            for row in self.edges
            if row.edge_kind == "CLAIM_PRESENTED_WORK"
        ]
        adjudicated = [
            row
            for row in self.edges
            if row.edge_kind == "WORK_ADJUDICATED_TO_CLAIM"
        ]
        unclassified = [
            row
            for row in presented
            if row.review_status == "UNCLASSIFIED"
        ]

        if self.presented_work_edge_count != len(presented):
            raise ValueError(
                "presented_work_edge_count mismatch"
            )
        if self.adjudicated_relation_edge_count != len(
            adjudicated
        ):
            raise ValueError(
                "adjudicated_relation_edge_count mismatch"
            )
        if self.unclassified_presentation_edge_count != len(
            unclassified
        ):
            raise ValueError(
                "unclassified_presentation_edge_count mismatch"
            )

        # A work omitted by the adjudicator for a claim may be represented as
        # presentation coverage only. It must never silently become a negative
        # or classified relation edge in this graph.
        adjudicated_pairs = {
            (row.claim_id, row.source_node_id)
            for row in adjudicated
        }
        for row in unclassified:
            if (row.claim_id, row.target_node_id) in adjudicated_pairs:
                raise ValueError(
                    "unclassified presentation also has adjudication edge"
                )

        body = self.model_dump(mode="json")
        observed_id = body.pop("graph_id")
        observed_sha = body.pop("graph_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("claim evidence graph SHA mismatch")
        if observed_id != (
            "scientific_claim_evidence_graph:"
            + expected_sha[:20]
        ):
            raise ValueError("claim evidence graph ID mismatch")
        return self


def _concept_node_key(
    concept: ScientificConceptIR,
) -> tuple[str, tuple[str, ...]]:
    return (
        concept.normalized_text,
        tuple(sorted(set(concept.type_labels))),
    )


def _relation_concepts(
    relation: ScientificRelationIR,
) -> list[ScientificConceptIR]:
    return [
        *relation.endpoint_concepts,
        *relation.identity_concepts,
        *relation.scope_qualifiers,
        *relation.directional_qualifiers,
        relation.observable_concept,
    ]


def _projection_map(
    report: GroundedFactorProjectionReport,
) -> dict[str, GroundedFactorRelationProjection]:
    return {
        projection.projection_id: projection
        for projection_set in report.projection_sets
        for projection in projection_set.projections
    }


def build_scientific_claim_evidence_graph(
    *,
    relation_ir_report: ScientificRelationIRReport,
    projection_report: GroundedFactorProjectionReport,
    candidate_report: ProjectionRelationCandidateReport,
    adjudication_report: ProjectionRelationAdjudicationReport,
) -> ScientificClaimEvidenceGraphReport:
    if (
        projection_report.source_relation_ir_report_id
        != relation_ir_report.report_id
    ):
        raise ValueError(
            "projection report does not derive from supplied relation IR"
        )
    if (
        candidate_report.source_relation_ir_report_id
        != relation_ir_report.report_id
    ):
        raise ValueError(
            "candidate report does not derive from supplied relation IR"
        )
    if (
        candidate_report.source_projection_report_id
        != projection_report.report_id
    ):
        raise ValueError(
            "candidate report does not derive from supplied projection report"
        )
    if (
        adjudication_report.source_candidate_report_id
        != candidate_report.report_id
    ):
        raise ValueError(
            "adjudication report does not derive from supplied candidate report"
        )

    relations = {
        row.claim_id: row
        for row in relation_ir_report.relations
    }
    candidate_sets = {
        row.claim_id: row
        for row in candidate_report.claims
    }
    reviews = {
        row.claim_id: row
        for row in adjudication_report.reviews
    }
    projections = _projection_map(projection_report)

    unknown_candidate_claims = set(candidate_sets) - set(relations)
    unknown_review_claims = set(reviews) - set(relations)
    if unknown_candidate_claims:
        raise ValueError(
            "candidate report contains unknown relation claim IDs: "
            + repr(sorted(unknown_candidate_claims))
        )
    if unknown_review_claims:
        raise ValueError(
            "adjudication report contains unknown relation claim IDs: "
            + repr(sorted(unknown_review_claims))
        )
    if set(candidate_sets) != set(reviews):
        raise ValueError(
            "candidate/adjudication claim sets must match"
        )

    nodes_by_id: dict[str, ScientificClaimEvidenceGraphNode] = {}
    edges_by_id: dict[str, ScientificClaimEvidenceGraphEdge] = {}

    def add_node(
        node: ScientificClaimEvidenceGraphNode,
    ) -> None:
        existing = nodes_by_id.get(node.node_id)
        if existing is None:
            nodes_by_id[node.node_id] = node
            return
        if existing.model_dump(mode="json") != node.model_dump(
            mode="json"
        ):
            raise ValueError(
                "conflicting duplicate graph node: "
                + node.node_id
            )

    def add_edge(
        edge: ScientificClaimEvidenceGraphEdge,
    ) -> None:
        existing = edges_by_id.get(edge.edge_id)
        if existing is None:
            edges_by_id[edge.edge_id] = edge
            return
        if existing.model_dump(mode="json") != edge.model_dump(
            mode="json"
        ):
            raise ValueError(
                "conflicting duplicate graph edge: "
                + edge.edge_id
            )

    hypothesis_claims: dict[str, list[str]] = defaultdict(list)
    for relation in relation_ir_report.relations:
        hypothesis_claims[relation.hypothesis_id].append(
            relation.claim_id
        )

    # HYPOTHESIS nodes.
    for hypothesis_id in sorted(hypothesis_claims):
        add_node(
            ScientificClaimEvidenceGraphNode(
                node_id=_stable_id(
                    "scientific_claim_evidence_hypothesis",
                    hypothesis_id,
                ),
                node_kind="HYPOTHESIS",
                label=hypothesis_id,
                hypothesis_ids=[hypothesis_id],
                source_ids=[hypothesis_id],
            )
        )

    # Exact normalized concept nodes may be shared across claims. This is the
    # only cross-claim coalescing step and performs no synonym inference.
    concept_accumulator: dict[
        tuple[str, tuple[str, ...]],
        dict[str, object],
    ] = {}
    for relation in relation_ir_report.relations:
        for concept in _relation_concepts(relation):
            key = _concept_node_key(concept)
            row = concept_accumulator.setdefault(
                key,
                {
                    "labels": [],
                    "hypothesis_ids": [],
                    "claim_ids": [],
                    "source_ids": [],
                    "roles": [],
                    "type_labels": list(key[1]),
                },
            )
            row["labels"].append(concept.surface_text)
            row["hypothesis_ids"].append(
                relation.hypothesis_id
            )
            row["claim_ids"].append(relation.claim_id)
            row["source_ids"].append(concept.concept_id)
            row["roles"].append(concept.role)

    concept_node_ids: dict[
        tuple[str, tuple[str, ...]],
        str,
    ] = {}
    for key in sorted(concept_accumulator):
        normalized_text, type_key = key
        node_id = _stable_id(
            "scientific_claim_evidence_concept",
            normalized_text,
            *type_key,
        )
        concept_node_ids[key] = node_id
        row = concept_accumulator[key]
        add_node(
            ScientificClaimEvidenceGraphNode(
                node_id=node_id,
                node_kind="CONCEPT",
                label=_unique(row["labels"])[0],
                hypothesis_ids=sorted(
                    set(row["hypothesis_ids"])
                ),
                claim_ids=sorted(set(row["claim_ids"])),
                source_ids=sorted(set(row["source_ids"])),
                normalized_text=normalized_text,
                concept_roles=sorted(set(row["roles"])),
                type_labels=list(type_key),
            )
        )

    projection_node_ids: dict[str, str] = {}
    for projection_id in sorted(projections):
        projection = projections[projection_id]
        node_id = _stable_id(
            "scientific_claim_evidence_projection",
            projection_id,
        )
        projection_node_ids[projection_id] = node_id
        add_node(
            ScientificClaimEvidenceGraphNode(
                node_id=node_id,
                node_kind="PROJECTION",
                label=projection.canonical_search_query,
                hypothesis_ids=[projection.hypothesis_id],
                claim_ids=[projection.claim_id],
                source_ids=[projection.projection_id],
                relation_ir_id=projection.relation_ir_id,
                projection_kind=projection.projection_kind,
                factor_order=projection.factor_order,
                retained_factor_ids=list(
                    projection.retained_factor_ids
                ),
                omitted_factor_ids=list(
                    projection.omitted_factor_ids
                ),
            )
        )

    work_node_ids: dict[str, str] = {}
    work_accumulator: dict[str, dict[str, object]] = {}

    for candidate_set in candidate_report.claims:
        for candidate in candidate_set.candidates:
            work_id = candidate.review_work_id
            row = work_accumulator.setdefault(
                work_id,
                {
                    "candidate": candidate,
                    "hypothesis_ids": [],
                    "claim_ids": [],
                    "source_work_ids": [],
                },
            )
            row["hypothesis_ids"].append(
                candidate_set.hypothesis_id
            )
            row["claim_ids"].append(candidate_set.claim_id)
            row["source_work_ids"].extend(
                candidate.source_work_ids
            )

            current = row["candidate"]
            if (
                len(str(candidate.abstract or ""))
                > len(str(current.abstract or ""))
            ):
                row["candidate"] = candidate

    for work_id in sorted(work_accumulator):
        row = work_accumulator[work_id]
        candidate = row["candidate"]
        node_id = _stable_id(
            "scientific_claim_evidence_work",
            work_id,
        )
        work_node_ids[work_id] = node_id
        add_node(
            ScientificClaimEvidenceGraphNode(
                node_id=node_id,
                node_kind="WORK",
                label=candidate.title,
                hypothesis_ids=sorted(
                    set(row["hypothesis_ids"])
                ),
                claim_ids=sorted(set(row["claim_ids"])),
                source_ids=[work_id],
                title=candidate.title,
                year=candidate.year,
                doi=candidate.doi,
                url=candidate.url,
                source_work_ids=sorted(
                    set(row["source_work_ids"])
                ),
            )
        )

    claim_summaries: list[
        ScientificClaimEvidenceGraphClaimSummary
    ] = []

    for claim_id in sorted(relations):
        relation = relations[claim_id]
        hypothesis_node_id = _stable_id(
            "scientific_claim_evidence_hypothesis",
            relation.hypothesis_id,
        )
        claim_node_id = _stable_id(
            "scientific_claim_evidence_claim",
            claim_id,
        )
        review = reviews.get(claim_id)

        add_node(
            ScientificClaimEvidenceGraphNode(
                node_id=claim_node_id,
                node_kind="CLAIM",
                label=relation.claim_text,
                hypothesis_ids=[relation.hypothesis_id],
                claim_ids=[claim_id],
                source_ids=[claim_id],
                relation_ir_id=relation.relation_ir_id,
                claim_kind=relation.claim_kind,
                novelty_selection_role=(
                    relation.novelty_selection_role
                ),
                typing_status=relation.typing_status,
                relation_state=(
                    review.relation_state
                    if review is not None
                    else None
                ),
            )
        )
        add_edge(
            ScientificClaimEvidenceGraphEdge(
                edge_id=_stable_id(
                    "scientific_claim_evidence_edge",
                    "HYPOTHESIS_HAS_CLAIM",
                    relation.hypothesis_id,
                    claim_id,
                ),
                edge_kind="HYPOTHESIS_HAS_CLAIM",
                source_node_id=hypothesis_node_id,
                target_node_id=claim_node_id,
                hypothesis_id=relation.hypothesis_id,
                claim_id=claim_id,
            )
        )

        claim_concept_nodes: list[str] = []
        for concept in _relation_concepts(relation):
            key = _concept_node_key(concept)
            concept_node_id = concept_node_ids[key]
            claim_concept_nodes.append(concept_node_id)
            add_edge(
                ScientificClaimEvidenceGraphEdge(
                    edge_id=_stable_id(
                        "scientific_claim_evidence_edge",
                        "CLAIM_HAS_CONCEPT",
                        claim_id,
                        concept_node_id,
                        concept.role,
                        concept.concept_id,
                    ),
                    edge_kind="CLAIM_HAS_CONCEPT",
                    source_node_id=claim_node_id,
                    target_node_id=concept_node_id,
                    hypothesis_id=relation.hypothesis_id,
                    claim_id=claim_id,
                    concept_role=concept.role,
                )
            )

        claim_projection_nodes: list[str] = []
        for projection in sorted(
            (
                row
                for row in projections.values()
                if row.claim_id == claim_id
            ),
            key=lambda row: (
                row.factor_order,
                row.projection_kind,
                row.projection_id,
            ),
        ):
            projection_node_id = projection_node_ids[
                projection.projection_id
            ]
            claim_projection_nodes.append(
                projection_node_id
            )
            add_edge(
                ScientificClaimEvidenceGraphEdge(
                    edge_id=_stable_id(
                        "scientific_claim_evidence_edge",
                        "CLAIM_HAS_PROJECTION",
                        claim_id,
                        projection.projection_id,
                    ),
                    edge_kind="CLAIM_HAS_PROJECTION",
                    source_node_id=claim_node_id,
                    target_node_id=projection_node_id,
                    hypothesis_id=relation.hypothesis_id,
                    claim_id=claim_id,
                )
            )

        candidate_set = candidate_sets.get(claim_id)
        presented_work_nodes: list[str] = []
        classified_work_nodes: list[str] = []
        unclassified_work_nodes: list[str] = []
        relationship_counts: dict[str, int] = defaultdict(int)

        matches_by_id = (
            {
                row.work_id: row
                for row in review.matches
            }
            if review is not None
            else {}
        )

        if candidate_set is not None:
            for candidate in candidate_set.candidates:
                work_node_id = work_node_ids[
                    candidate.review_work_id
                ]
                presented_work_nodes.append(work_node_id)
                classified = (
                    candidate.review_work_id
                    in matches_by_id
                )
                if classified:
                    classified_work_nodes.append(
                        work_node_id
                    )
                else:
                    unclassified_work_nodes.append(
                        work_node_id
                    )

                add_edge(
                    ScientificClaimEvidenceGraphEdge(
                        edge_id=_stable_id(
                            "scientific_claim_evidence_edge",
                            "CLAIM_PRESENTED_WORK",
                            claim_id,
                            candidate.review_work_id,
                        ),
                        edge_kind="CLAIM_PRESENTED_WORK",
                        source_node_id=claim_node_id,
                        target_node_id=work_node_id,
                        hypothesis_id=relation.hypothesis_id,
                        claim_id=claim_id,
                        review_status=(
                            "CLASSIFIED"
                            if classified
                            else "UNCLASSIFIED"
                        ),
                        typed_compatibility_state=(
                            candidate.typed_compatibility_state
                        ),
                        selected_lanes=list(
                            candidate.selected_lanes
                        ),
                        relation_anchor_tier=(
                            candidate.relation_anchor_tier
                        ),
                        endpoint_supported_count=(
                            candidate.endpoint_supported_count
                        ),
                        second_pass_roles=list(
                            candidate.second_pass_roles
                        ),
                        counterevidence_modes=list(
                            candidate.counterevidence_modes
                        ),
                    )
                )

        if review is not None:
            for match in review.matches:
                work_node_id = work_node_ids.get(
                    match.work_id
                )
                if work_node_id is None:
                    raise ValueError(
                        "adjudicated work missing from candidate graph: "
                        + match.work_id
                    )

                relationship_counts[
                    match.relationship
                ] += 1

                add_edge(
                    ScientificClaimEvidenceGraphEdge(
                        edge_id=_stable_id(
                            "scientific_claim_evidence_edge",
                            "WORK_ADJUDICATED_TO_CLAIM",
                            claim_id,
                            match.work_id,
                            match.relationship,
                        ),
                        edge_kind="WORK_ADJUDICATED_TO_CLAIM",
                        source_node_id=work_node_id,
                        target_node_id=claim_node_id,
                        hypothesis_id=relation.hypothesis_id,
                        claim_id=claim_id,
                        review_status="CLASSIFIED",
                        typed_compatibility_state=(
                            match.typed_compatibility_state
                        ),
                        second_pass_roles=list(
                            match.second_pass_roles
                        ),
                        counterevidence_modes=list(
                            match.counterevidence_modes
                        ),
                        relationship=match.relationship,
                        original_relationship=(
                            match.original_relationship
                        ),
                        confidence=match.confidence,
                        evidence_span=match.evidence_span,
                        rationale=match.rationale,
                        deterministic_reason_codes=list(
                            match.deterministic_reason_codes
                        ),
                    )
                )

                for projection_id in match.basis_projection_ids:
                    projection_node_id = (
                        projection_node_ids.get(
                            projection_id
                        )
                    )
                    if projection_node_id is None:
                        raise ValueError(
                            "adjudication basis projection missing "
                            "from supplied projection report: "
                            + projection_id
                        )
                    add_edge(
                        ScientificClaimEvidenceGraphEdge(
                            edge_id=_stable_id(
                                "scientific_claim_evidence_edge",
                                "WORK_BASIS_PROJECTION",
                                claim_id,
                                match.work_id,
                                projection_id,
                            ),
                            edge_kind="WORK_BASIS_PROJECTION",
                            source_node_id=work_node_id,
                            target_node_id=projection_node_id,
                            hypothesis_id=relation.hypothesis_id,
                            claim_id=claim_id,
                            review_status="CLASSIFIED",
                            relationship=match.relationship,
                            basis_projection_id=projection_id,
                        )
                    )

        claim_summaries.append(
            ScientificClaimEvidenceGraphClaimSummary(
                hypothesis_id=relation.hypothesis_id,
                claim_id=claim_id,
                relation_ir_id=relation.relation_ir_id,
                claim_kind=relation.claim_kind,
                novelty_selection_role=(
                    relation.novelty_selection_role
                ),
                typing_status=relation.typing_status,
                adjudication_status=(
                    "REVIEWED"
                    if review is not None
                    else "NOT_REVIEWED"
                ),
                relation_state=(
                    review.relation_state
                    if review is not None
                    else None
                ),
                concept_node_ids=sorted(
                    set(claim_concept_nodes)
                ),
                projection_node_ids=sorted(
                    set(claim_projection_nodes)
                ),
                presented_work_node_ids=sorted(
                    set(presented_work_nodes)
                ),
                classified_work_node_ids=sorted(
                    set(classified_work_nodes)
                ),
                unclassified_work_node_ids=sorted(
                    set(unclassified_work_nodes)
                ),
                presented_work_count=len(
                    set(presented_work_nodes)
                ),
                classified_work_count=len(
                    set(classified_work_nodes)
                ),
                unclassified_work_count=len(
                    set(unclassified_work_nodes)
                ),
                relationship_counts=dict(
                    sorted(relationship_counts.items())
                ),
            )
        )

    summaries_by_hypothesis: dict[
        str,
        list[ScientificClaimEvidenceGraphClaimSummary],
    ] = defaultdict(list)
    for summary in claim_summaries:
        summaries_by_hypothesis[
            summary.hypothesis_id
        ].append(summary)

    hypothesis_summaries: list[
        ScientificClaimEvidenceGraphHypothesisSummary
    ] = []
    for hypothesis_id in sorted(summaries_by_hypothesis):
        rows = sorted(
            summaries_by_hypothesis[hypothesis_id],
            key=lambda row: row.claim_id,
        )
        reviewed_ids = [
            row.claim_id
            for row in rows
            if row.adjudication_status == "REVIEWED"
        ]
        unreviewed_ids = [
            row.claim_id
            for row in rows
            if row.adjudication_status == "NOT_REVIEWED"
        ]
        hypothesis_summaries.append(
            ScientificClaimEvidenceGraphHypothesisSummary(
                hypothesis_id=hypothesis_id,
                claim_ids=[row.claim_id for row in rows],
                novelty_bearing_claim_ids=[
                    row.claim_id
                    for row in rows
                    if row.novelty_selection_role
                    == "NOVELTY_BEARING"
                ],
                reviewed_claim_ids=reviewed_ids,
                unreviewed_claim_ids=unreviewed_ids,
                claim_count=len(rows),
                reviewed_claim_count=len(reviewed_ids),
                unreviewed_claim_count=len(
                    unreviewed_ids
                ),
            )
        )

    nodes = sorted(
        nodes_by_id.values(),
        key=lambda row: (
            row.node_kind,
            row.node_id,
        ),
    )
    edges = sorted(
        edges_by_id.values(),
        key=lambda row: (
            row.edge_kind,
            row.claim_id or "",
            row.edge_id,
        ),
    )

    body = {
        "schema_version":
            "scientific-claim-evidence-graph-report-v1",
        "source_relation_ir_report_id":
            relation_ir_report.report_id,
        "source_projection_report_id":
            projection_report.report_id,
        "source_candidate_report_id":
            candidate_report.report_id,
        "source_adjudication_report_id":
            adjudication_report.report_id,
        "nodes": [
            row.model_dump(mode="json")
            for row in nodes
        ],
        "edges": [
            row.model_dump(mode="json")
            for row in edges
        ],
        "claim_summaries": [
            row.model_dump(mode="json")
            for row in sorted(
                claim_summaries,
                key=lambda row: row.claim_id,
            )
        ],
        "hypothesis_summaries": [
            row.model_dump(mode="json")
            for row in hypothesis_summaries
        ],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "hypothesis_count": len(
            hypothesis_summaries
        ),
        "claim_count": len(claim_summaries),
        "concept_node_count": sum(
            row.node_kind == "CONCEPT"
            for row in nodes
        ),
        "projection_node_count": sum(
            row.node_kind == "PROJECTION"
            for row in nodes
        ),
        "work_node_count": sum(
            row.node_kind == "WORK"
            for row in nodes
        ),
        "presented_work_edge_count": sum(
            row.edge_kind == "CLAIM_PRESENTED_WORK"
            for row in edges
        ),
        "adjudicated_relation_edge_count": sum(
            row.edge_kind
            == "WORK_ADJUDICATED_TO_CLAIM"
            for row in edges
        ),
        "unclassified_presentation_edge_count": sum(
            row.edge_kind == "CLAIM_PRESENTED_WORK"
            and row.review_status == "UNCLASSIFIED"
            for row in edges
        ),
        "graph_semantics":
            "deterministic_claim_evidence_topology_v1",
        "exact_normalized_concept_coalescing_only":
            True,
        "classified_and_unclassified_review_separated":
            True,
        "unclassified_is_not_negative_evidence":
            True,
        "absence_based_novelty_authorized":
            False,
        "positive_nonobviousness_authority_created":
            False,
        "centrality_scoring_performed":
            False,
        "aggregation_performed":
            False,
        "diagnostic_only":
            True,
        "n9_contract_changed":
            False,
        "n10_contract_changed":
            False,
        "production_selection_changed":
            False,
        "canonical_graph_mutated":
            False,
    }
    digest = _sha256_json(body)

    return ScientificClaimEvidenceGraphReport(
        **body,
        graph_id=(
            "scientific_claim_evidence_graph:"
            + digest[:20]
        ),
        graph_sha256=digest,
    )


__all__ = [
    "ScientificClaimEvidenceGraphClaimSummary",
    "ScientificClaimEvidenceGraphEdge",
    "ScientificClaimEvidenceGraphHypothesisSummary",
    "ScientificClaimEvidenceGraphNode",
    "ScientificClaimEvidenceGraphReport",
    "build_scientific_claim_evidence_graph",
]
