from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphClaimSummary,
    ScientificClaimEvidenceGraphReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ClaimEvidencePressureState = Literal[
    "DIRECT_PRIOR_ART_PRESSURE",
    "LOWER_ORDER_PRIOR_ART_PRESSURE",
    "COUNTEREVIDENCE_PRESSURE",
    "CONFLICTING_EVIDENCE_PRESSURE",
    "MIXED_STRONG_SIGNAL_PRESSURE",
    "PARTIAL_PRIOR_ART_PRESSURE",
    "COMPONENT_CONTEXT_ONLY",
    "NO_MATERIAL_SIGNAL_IN_CLASSIFIED_SUBSET",
    "INSUFFICIENT_METADATA",
    "NOT_REVIEWED",
]

TopologyState = Literal[
    "SINGLE_CLAIM_DEGENERATE",
    "MULTI_CLAIM_CONNECTED",
    "MULTI_CLAIM_NO_EXACT_SHARED_CONCEPTS",
]

ClassificationCoverageState = Literal[
    "NO_PRESENTED_WORKS",
    "FULLY_CLASSIFIED",
    "PARTIALLY_CLASSIFIED",
    "UNCLASSIFIED_ONLY",
]

_ROLE_PRECEDENCE = {
    "NOVELTY_BEARING": 0,
    "REQUIRED_ENABLING_RELATION": 1,
    "TESTING_PREDICTION": 2,
    "AUXILIARY": 3,
}

_STRONG_PRESSURE_STATES = frozenset(
    {
        "DIRECT_PRIOR_ART_PRESSURE",
        "LOWER_ORDER_PRIOR_ART_PRESSURE",
        "COUNTEREVIDENCE_PRESSURE",
        "CONFLICTING_EVIDENCE_PRESSURE",
        "MIXED_STRONG_SIGNAL_PRESSURE",
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


def _jaccard(
    left: set[str],
    right: set[str],
) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _coverage_state(
    row: ScientificClaimEvidenceGraphClaimSummary,
) -> ClassificationCoverageState:
    if row.presented_work_count == 0:
        return "NO_PRESENTED_WORKS"
    if row.classified_work_count == row.presented_work_count:
        return "FULLY_CLASSIFIED"
    if row.classified_work_count == 0:
        return "UNCLASSIFIED_ONLY"
    return "PARTIALLY_CLASSIFIED"


def _pressure_state(
    row: ScientificClaimEvidenceGraphClaimSummary,
) -> ClaimEvidencePressureState:
    if row.adjudication_status != "REVIEWED":
        return "NOT_REVIEWED"

    counts = row.relationship_counts

    direct = counts.get("DIRECT_PRIOR_ART", 0) > 0
    lower = counts.get(
        "LOWER_ORDER_RELATION_PRIOR_ART",
        0,
    ) > 0
    counter = (
        counts.get("DIRECTIONAL_COUNTEREVIDENCE", 0) > 0
        or counts.get("CONTEXTUAL_CONFLICT", 0) > 0
    )
    conflicting = counts.get(
        "CONFLICTING_PRIOR_ART",
        0,
    ) > 0

    strong_count = sum(
        (direct, lower, counter, conflicting)
    )
    if strong_count >= 2:
        return "MIXED_STRONG_SIGNAL_PRESSURE"
    if direct:
        return "DIRECT_PRIOR_ART_PRESSURE"
    if lower:
        return "LOWER_ORDER_PRIOR_ART_PRESSURE"
    if conflicting:
        return "CONFLICTING_EVIDENCE_PRESSURE"
    if counter:
        return "COUNTEREVIDENCE_PRESSURE"

    if counts.get("PARTIAL_PRIOR_ART", 0) > 0:
        return "PARTIAL_PRIOR_ART_PRESSURE"
    if counts.get("COMPONENT_ONLY", 0) > 0:
        return "COMPONENT_CONTEXT_ONLY"

    if row.relation_state == "INSUFFICIENT_METADATA":
        return "INSUFFICIENT_METADATA"

    return "NO_MATERIAL_SIGNAL_IN_CLASSIFIED_SUBSET"


class ScientificClaimPairTopology(StrictModel):
    hypothesis_id: str
    left_claim_id: str
    right_claim_id: str

    left_concept_count: int = Field(ge=0)
    right_concept_count: int = Field(ge=0)
    shared_concept_node_ids: list[str]
    shared_concept_count: int = Field(ge=0)

    concept_jaccard: float = Field(ge=0.0, le=1.0)
    exact_shared_concept_edge: bool

    @model_validator(mode="after")
    def validate_pair(self) -> "ScientificClaimPairTopology":
        if self.left_claim_id >= self.right_claim_id:
            raise ValueError(
                "claim pair IDs must be stored in sorted order"
            )
        if self.shared_concept_count != len(
            self.shared_concept_node_ids
        ):
            raise ValueError("shared concept count mismatch")
        if self.exact_shared_concept_edge != (
            self.shared_concept_count > 0
        ):
            raise ValueError(
                "exact_shared_concept_edge mismatch"
            )
        return self


class ScientificClaimCentralityRecord(StrictModel):
    hypothesis_id: str
    claim_id: str
    novelty_selection_role: str
    role_precedence: int = Field(ge=0)
    claim_kind: str
    typing_status: str

    topology_state: TopologyState
    hypothesis_claim_count: int = Field(ge=1)
    concept_node_count: int = Field(ge=0)

    neighbor_claim_count: int = Field(ge=0)
    neighbor_degree_centrality: float = Field(
        ge=0.0,
        le=1.0,
    )
    shared_concept_node_ids: list[str]
    shared_concept_count: int = Field(ge=0)
    shared_concept_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    mean_pairwise_concept_jaccard: float = Field(
        ge=0.0,
        le=1.0,
    )

    # This is a graph-topology statistic only. It deliberately does not
    # combine role semantics or evidence labels into one weighted score.
    structural_centrality_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    structural_centrality_definition: Literal[
        "mean_pairwise_exact_concept_jaccard_v1"
    ] = "mean_pairwise_exact_concept_jaccard_v1"

    centrality_rank_within_hypothesis: int = Field(ge=1)
    centrality_rank_tied: bool = False

    classification_coverage_state: ClassificationCoverageState
    classification_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    evidence_pressure_state: ClaimEvidencePressureState
    strong_evidence_pressure_present: bool

    # The evidence-pressure label is computed only from emitted adjudication
    # records. It is not a complete literature judgment when unclassified
    # presented works remain.
    evidence_pressure_is_bounded_to_classified_subset: Literal[
        True
    ] = True

    unclassified_is_not_negative_evidence: Literal[True] = True
    absence_based_novelty_authorized: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "ScientificClaimCentralityRecord":
        if self.shared_concept_count != len(
            self.shared_concept_node_ids
        ):
            raise ValueError(
                "centrality shared concept count mismatch"
            )
        if self.hypothesis_claim_count == 1:
            if self.topology_state != "SINGLE_CLAIM_DEGENERATE":
                raise ValueError(
                    "single claim hypothesis must be degenerate"
                )
            if self.structural_centrality_score != 1.0:
                raise ValueError(
                    "single claim centrality convention must be 1.0"
                )
        return self


class ScientificHypothesisCentralitySummary(StrictModel):
    hypothesis_id: str
    topology_state: TopologyState

    claim_ids: list[str]
    novelty_bearing_claim_ids: list[str]
    role_ordered_claim_ids: list[str]
    centrality_ordered_claim_ids: list[str]

    claim_count: int = Field(ge=1)
    exact_shared_claim_pair_count: int = Field(ge=0)
    total_claim_pair_count: int = Field(ge=0)

    classification_coverage_states: dict[str, int]
    evidence_pressure_states: dict[str, int]
    strong_pressure_claim_ids: list[str]

    all_claims_fully_classified: bool
    any_unclassified_presented_work: bool
    centrality_is_degenerate: bool

    # No overall novelty verdict is produced here.
    aggregation_readiness: Literal[
        "STRUCTURAL_CENTRALITY_READY_EPISTEMICALLY_PARTIAL",
        "STRUCTURAL_CENTRALITY_READY_FULL_CLASSIFICATION",
    ]

    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "ScientificHypothesisCentralitySummary":
        if self.claim_count != len(self.claim_ids):
            raise ValueError(
                "hypothesis centrality claim_count mismatch"
            )
        expected_pairs = (
            self.claim_count * (self.claim_count - 1) // 2
        )
        if self.total_claim_pair_count != expected_pairs:
            raise ValueError(
                "hypothesis pair count mismatch"
            )
        if (
            self.exact_shared_claim_pair_count
            > self.total_claim_pair_count
        ):
            raise ValueError(
                "shared pair count exceeds total pairs"
            )
        return self


class ScientificClaimCentralityReport(StrictModel):
    schema_version: Literal[
        "scientific-claim-centrality-report-v1"
    ] = "scientific-claim-centrality-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_claim_evidence_graph_id: str

    claim_records: list[ScientificClaimCentralityRecord]
    pair_topologies: list[ScientificClaimPairTopology]
    hypothesis_summaries: list[
        ScientificHypothesisCentralitySummary
    ]

    hypothesis_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    pair_topology_count: int = Field(ge=0)
    single_claim_hypothesis_count: int = Field(ge=0)
    multi_claim_connected_hypothesis_count: int = Field(ge=0)
    multi_claim_no_shared_concept_hypothesis_count: int = Field(ge=0)

    centrality_semantics: Literal[
        "exact_shared_concept_topology_only_v1"
    ] = "exact_shared_concept_topology_only_v1"

    role_semantics_kept_separate_from_centrality: Literal[True] = True
    evidence_pressure_kept_separate_from_centrality: Literal[True] = True
    synonym_inference_performed: Literal[False] = False
    unclassified_is_not_negative_evidence: Literal[True] = True
    absence_based_novelty_authorized: Literal[False] = False
    positive_nonobviousness_authority_created: Literal[False] = False
    hypothesis_novelty_aggregation_performed: Literal[False] = False
    diagnostic_only: Literal[True] = True
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ScientificClaimCentralityReport":
        if self.claim_count != len(self.claim_records):
            raise ValueError("centrality claim_count mismatch")
        if self.hypothesis_count != len(
            self.hypothesis_summaries
        ):
            raise ValueError(
                "centrality hypothesis_count mismatch"
            )
        if self.pair_topology_count != len(
            self.pair_topologies
        ):
            raise ValueError(
                "centrality pair_topology_count mismatch"
            )

        states = [
            row.topology_state
            for row in self.hypothesis_summaries
        ]
        if self.single_claim_hypothesis_count != states.count(
            "SINGLE_CLAIM_DEGENERATE"
        ):
            raise ValueError(
                "single claim hypothesis count mismatch"
            )
        if (
            self.multi_claim_connected_hypothesis_count
            != states.count("MULTI_CLAIM_CONNECTED")
        ):
            raise ValueError(
                "multi-claim connected count mismatch"
            )
        if (
            self.multi_claim_no_shared_concept_hypothesis_count
            != states.count(
                "MULTI_CLAIM_NO_EXACT_SHARED_CONCEPTS"
            )
        ):
            raise ValueError(
                "multi-claim disconnected count mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "claim centrality report SHA mismatch"
            )
        if observed_id != (
            "scientific_claim_centrality:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "claim centrality report ID mismatch"
            )
        return self


def _topology_state(
    *,
    claim_count: int,
    shared_pair_count: int,
) -> TopologyState:
    if claim_count == 1:
        return "SINGLE_CLAIM_DEGENERATE"
    if shared_pair_count > 0:
        return "MULTI_CLAIM_CONNECTED"
    return "MULTI_CLAIM_NO_EXACT_SHARED_CONCEPTS"


def build_scientific_claim_centrality_report(
    graph: ScientificClaimEvidenceGraphReport,
) -> ScientificClaimCentralityReport:
    claim_by_id = {
        row.claim_id: row
        for row in graph.claim_summaries
    }
    hypothesis_by_id = {
        row.hypothesis_id: row
        for row in graph.hypothesis_summaries
    }

    pair_topologies: list[ScientificClaimPairTopology] = []
    claim_records: list[ScientificClaimCentralityRecord] = []
    hypothesis_summaries: list[
        ScientificHypothesisCentralitySummary
    ] = []

    for hypothesis_id in sorted(hypothesis_by_id):
        hypothesis = hypothesis_by_id[hypothesis_id]
        claim_ids = sorted(hypothesis.claim_ids)
        claims = [
            claim_by_id[claim_id]
            for claim_id in claim_ids
        ]
        claim_count = len(claims)
        if claim_count < 1:
            raise ValueError(
                "hypothesis centrality requires at least one claim"
            )

        concept_sets = {
            row.claim_id: set(row.concept_node_ids)
            for row in claims
        }

        jaccards_by_claim: dict[str, list[float]] = {
            claim_id: []
            for claim_id in claim_ids
        }
        neighbors_by_claim: dict[str, set[str]] = {
            claim_id: set()
            for claim_id in claim_ids
        }
        shared_concepts_by_claim: dict[str, set[str]] = {
            claim_id: set()
            for claim_id in claim_ids
        }

        shared_pair_count = 0
        for left_index, left_id in enumerate(claim_ids):
            for right_id in claim_ids[left_index + 1 :]:
                left = concept_sets[left_id]
                right = concept_sets[right_id]
                shared = sorted(left & right)
                jaccard = _jaccard(left, right)

                if shared:
                    shared_pair_count += 1
                    neighbors_by_claim[left_id].add(right_id)
                    neighbors_by_claim[right_id].add(left_id)
                    shared_concepts_by_claim[left_id].update(
                        shared
                    )
                    shared_concepts_by_claim[right_id].update(
                        shared
                    )

                jaccards_by_claim[left_id].append(jaccard)
                jaccards_by_claim[right_id].append(jaccard)

                pair_topologies.append(
                    ScientificClaimPairTopology(
                        hypothesis_id=hypothesis_id,
                        left_claim_id=left_id,
                        right_claim_id=right_id,
                        left_concept_count=len(left),
                        right_concept_count=len(right),
                        shared_concept_node_ids=shared,
                        shared_concept_count=len(shared),
                        concept_jaccard=jaccard,
                        exact_shared_concept_edge=bool(shared),
                    )
                )

        topology_state = _topology_state(
            claim_count=claim_count,
            shared_pair_count=shared_pair_count,
        )

        provisional: list[dict[str, object]] = []
        for row in claims:
            concept_set = concept_sets[row.claim_id]
            shared = shared_concepts_by_claim[
                row.claim_id
            ]
            neighbor_count = len(
                neighbors_by_claim[row.claim_id]
            )

            if claim_count == 1:
                degree = 1.0
                mean_jaccard = 1.0
                shared_fraction = 1.0
            else:
                degree = neighbor_count / (claim_count - 1)
                values = jaccards_by_claim[row.claim_id]
                mean_jaccard = (
                    sum(values) / len(values)
                    if values
                    else 0.0
                )
                shared_fraction = (
                    len(shared) / len(concept_set)
                    if concept_set
                    else 0.0
                )

            coverage_state = _coverage_state(row)
            classification_fraction = (
                row.classified_work_count
                / row.presented_work_count
                if row.presented_work_count
                else 0.0
            )
            pressure = _pressure_state(row)

            provisional.append(
                {
                    "summary": row,
                    "role_precedence":
                        _ROLE_PRECEDENCE.get(
                            row.novelty_selection_role,
                            len(_ROLE_PRECEDENCE),
                        ),
                    "degree": degree,
                    "mean_jaccard": mean_jaccard,
                    "shared": sorted(shared),
                    "shared_fraction": shared_fraction,
                    "coverage_state": coverage_state,
                    "classification_fraction":
                        classification_fraction,
                    "pressure": pressure,
                }
            )

        # Rank only by structural centrality. Role precedence is deliberately
        # excluded so that role semantics cannot masquerade as graph centrality.
        ranked = sorted(
            provisional,
            key=lambda item: (
                -float(item["mean_jaccard"]),
                -float(item["degree"]),
                -float(item["shared_fraction"]),
                str(item["summary"].claim_id),
            ),
        )

        rank_by_claim: dict[str, int] = {}
        tied_by_claim: dict[str, bool] = {}
        previous_key: tuple[float, float, float] | None = None
        previous_rank = 0
        rank_groups: dict[
            tuple[float, float, float],
            list[str],
        ] = defaultdict(list)

        for ordinal, item in enumerate(ranked, start=1):
            key = (
                float(item["mean_jaccard"]),
                float(item["degree"]),
                float(item["shared_fraction"]),
            )
            if key != previous_key:
                previous_rank = ordinal
                previous_key = key
            claim_id = item["summary"].claim_id
            rank_by_claim[claim_id] = previous_rank
            rank_groups[key].append(claim_id)

        for group in rank_groups.values():
            tied = len(group) > 1
            for claim_id in group:
                tied_by_claim[claim_id] = tied

        records_for_hypothesis: list[
            ScientificClaimCentralityRecord
        ] = []
        for item in provisional:
            row = item["summary"]
            record = ScientificClaimCentralityRecord(
                hypothesis_id=hypothesis_id,
                claim_id=row.claim_id,
                novelty_selection_role=(
                    row.novelty_selection_role
                ),
                role_precedence=int(
                    item["role_precedence"]
                ),
                claim_kind=row.claim_kind,
                typing_status=row.typing_status,
                topology_state=topology_state,
                hypothesis_claim_count=claim_count,
                concept_node_count=len(
                    concept_sets[row.claim_id]
                ),
                neighbor_claim_count=len(
                    neighbors_by_claim[row.claim_id]
                ),
                neighbor_degree_centrality=float(
                    item["degree"]
                ),
                shared_concept_node_ids=list(
                    item["shared"]
                ),
                shared_concept_count=len(
                    item["shared"]
                ),
                shared_concept_fraction=float(
                    item["shared_fraction"]
                ),
                mean_pairwise_concept_jaccard=float(
                    item["mean_jaccard"]
                ),
                structural_centrality_score=float(
                    item["mean_jaccard"]
                ),
                centrality_rank_within_hypothesis=(
                    rank_by_claim[row.claim_id]
                ),
                centrality_rank_tied=(
                    tied_by_claim[row.claim_id]
                ),
                classification_coverage_state=(
                    item["coverage_state"]
                ),
                classification_fraction=float(
                    item["classification_fraction"]
                ),
                evidence_pressure_state=(
                    item["pressure"]
                ),
                strong_evidence_pressure_present=(
                    item["pressure"]
                    in _STRONG_PRESSURE_STATES
                ),
            )
            records_for_hypothesis.append(record)
            claim_records.append(record)

        coverage_counts: dict[str, int] = defaultdict(int)
        pressure_counts: dict[str, int] = defaultdict(int)
        for record in records_for_hypothesis:
            coverage_counts[
                record.classification_coverage_state
            ] += 1
            pressure_counts[
                record.evidence_pressure_state
            ] += 1

        all_fully_classified = all(
            record.classification_coverage_state
            in {
                "FULLY_CLASSIFIED",
                "NO_PRESENTED_WORKS",
            }
            for record in records_for_hypothesis
        )
        any_unclassified = any(
            record.classification_coverage_state
            in {
                "PARTIALLY_CLASSIFIED",
                "UNCLASSIFIED_ONLY",
            }
            for record in records_for_hypothesis
        )

        role_ordered = sorted(
            records_for_hypothesis,
            key=lambda record: (
                record.role_precedence,
                record.centrality_rank_within_hypothesis,
                record.claim_id,
            ),
        )
        centrality_ordered = sorted(
            records_for_hypothesis,
            key=lambda record: (
                record.centrality_rank_within_hypothesis,
                record.role_precedence,
                record.claim_id,
            ),
        )

        hypothesis_summaries.append(
            ScientificHypothesisCentralitySummary(
                hypothesis_id=hypothesis_id,
                topology_state=topology_state,
                claim_ids=claim_ids,
                novelty_bearing_claim_ids=sorted(
                    hypothesis.novelty_bearing_claim_ids
                ),
                role_ordered_claim_ids=[
                    row.claim_id
                    for row in role_ordered
                ],
                centrality_ordered_claim_ids=[
                    row.claim_id
                    for row in centrality_ordered
                ],
                claim_count=claim_count,
                exact_shared_claim_pair_count=(
                    shared_pair_count
                ),
                total_claim_pair_count=(
                    claim_count
                    * (claim_count - 1)
                    // 2
                ),
                classification_coverage_states=dict(
                    sorted(coverage_counts.items())
                ),
                evidence_pressure_states=dict(
                    sorted(pressure_counts.items())
                ),
                strong_pressure_claim_ids=sorted(
                    record.claim_id
                    for record in records_for_hypothesis
                    if record.strong_evidence_pressure_present
                ),
                all_claims_fully_classified=(
                    all_fully_classified
                ),
                any_unclassified_presented_work=(
                    any_unclassified
                ),
                centrality_is_degenerate=(
                    topology_state
                    == "SINGLE_CLAIM_DEGENERATE"
                ),
                aggregation_readiness=(
                    "STRUCTURAL_CENTRALITY_READY_FULL_CLASSIFICATION"
                    if all_fully_classified
                    else
                    "STRUCTURAL_CENTRALITY_READY_EPISTEMICALLY_PARTIAL"
                ),
            )
        )

    claim_records.sort(
        key=lambda row: (
            row.hypothesis_id,
            row.centrality_rank_within_hypothesis,
            row.role_precedence,
            row.claim_id,
        )
    )
    pair_topologies.sort(
        key=lambda row: (
            row.hypothesis_id,
            row.left_claim_id,
            row.right_claim_id,
        )
    )
    hypothesis_summaries.sort(
        key=lambda row: row.hypothesis_id
    )

    body = {
        "schema_version":
            "scientific-claim-centrality-report-v1",
        "source_claim_evidence_graph_id":
            graph.graph_id,
        "claim_records": [
            row.model_dump(mode="json")
            for row in claim_records
        ],
        "pair_topologies": [
            row.model_dump(mode="json")
            for row in pair_topologies
        ],
        "hypothesis_summaries": [
            row.model_dump(mode="json")
            for row in hypothesis_summaries
        ],
        "hypothesis_count":
            len(hypothesis_summaries),
        "claim_count":
            len(claim_records),
        "pair_topology_count":
            len(pair_topologies),
        "single_claim_hypothesis_count":
            sum(
                row.topology_state
                == "SINGLE_CLAIM_DEGENERATE"
                for row in hypothesis_summaries
            ),
        "multi_claim_connected_hypothesis_count":
            sum(
                row.topology_state
                == "MULTI_CLAIM_CONNECTED"
                for row in hypothesis_summaries
            ),
        "multi_claim_no_shared_concept_hypothesis_count":
            sum(
                row.topology_state
                == "MULTI_CLAIM_NO_EXACT_SHARED_CONCEPTS"
                for row in hypothesis_summaries
            ),
        "centrality_semantics":
            "exact_shared_concept_topology_only_v1",
        "role_semantics_kept_separate_from_centrality":
            True,
        "evidence_pressure_kept_separate_from_centrality":
            True,
        "synonym_inference_performed":
            False,
        "unclassified_is_not_negative_evidence":
            True,
        "absence_based_novelty_authorized":
            False,
        "positive_nonobviousness_authority_created":
            False,
        "hypothesis_novelty_aggregation_performed":
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

    return ScientificClaimCentralityReport(
        **body,
        report_id=(
            "scientific_claim_centrality:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "ScientificClaimCentralityRecord",
    "ScientificClaimCentralityReport",
    "ScientificClaimPairTopology",
    "ScientificHypothesisCentralitySummary",
    "build_scientific_claim_centrality_report",
]
