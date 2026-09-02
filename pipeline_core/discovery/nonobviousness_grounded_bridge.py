from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.discovery_semantics import (
    is_alignment_edge,
    is_scaffold_edge,
)
from pipeline_core.discovery.explorer_contracts import (
    GraphExplorerPacket,
)
from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeOpportunity,
)
from pipeline_core.domain.domain_profile import (
    DiscoverySemantics,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class N11GroundedBridgeQuery(StrictModel):
    schema_version: Literal[
        "n11-grounded-bridge-query-v1"
    ] = "n11-grounded-bridge-query-v1"

    query_id: str = Field(min_length=1)

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    factor_identity_term: str = Field(
        min_length=1
    )

    base_context_term: str = Field(
        min_length=1
    )

    traversal_mode: Literal[
        "mechanism"
    ] = "mechanism"

    traversal_algorithm: Literal[
        "top_n"
    ] = "top_n"

    top_k: Literal[5] = 5
    max_depth: Literal[8] = 8

    # D2 must not use corpus-alignment edges to manufacture a bridge.
    max_alignment_edges: Literal[0] = 0
    min_scientific_edges: Literal[1] = 1

    allow_candidate_endpoints: Literal[
        False
    ] = False

    allow_alignment_hub_endpoints: Literal[
        False
    ] = False

    source_bounded_vocabulary_only: Literal[
        True
    ] = True

    production_authority: Literal[
        False
    ] = False


class N11GroundedBridgeQueryPlan(
    StrictModel
):
    schema_version: Literal[
        "n11-grounded-bridge-query-plan-v1"
    ] = "n11-grounded-bridge-query-plan-v1"

    plan_id: str = Field(min_length=1)

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    queries: list[
        N11GroundedBridgeQuery
    ] = Field(
        min_length=1
    )

    production_authority: Literal[
        False
    ] = False


class N11GroundedBridgeCandidate(
    StrictModel
):
    schema_version: Literal[
        "n11-grounded-bridge-candidate-v1"
    ] = "n11-grounded-bridge-candidate-v1"

    bridge_candidate_id: str = Field(
        min_length=1
    )

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    factor_node_ids: list[str] = Field(
        min_length=1
    )

    intermediate_node_ids: list[str]

    base_context_node_ids: list[str] = Field(
        min_length=1
    )

    packet_ids: list[str] = Field(
        min_length=1
    )

    path_ids: list[str] = Field(
        min_length=1
    )

    edge_ids: list[str] = Field(
        min_length=1
    )

    source_paper_ids: list[str] = Field(
        min_length=1
    )

    path_class: Literal[
        "DIRECT_SCIENTIFIC_CHAIN"
    ] = "DIRECT_SCIENTIFIC_CHAIN"

    scientific_relation_texts: list[str] = Field(
        min_length=1
    )

    grounded_edge_count: int = Field(
        ge=1
    )

    alignment_dependent: Literal[
        False
    ] = False

    common_anchor_only: Literal[
        False
    ] = False

    navigation_only: Literal[
        False
    ] = False

    eligible_for_operator_reconsideration: Literal[
        True
    ] = True

    production_authority: Literal[
        False
    ] = False

    @model_validator(
        mode="after"
    )
    def _edge_count_consistency(
        self,
    ) -> "N11GroundedBridgeCandidate":
        if self.grounded_edge_count != len(
            self.edge_ids
        ):
            raise ValueError(
                "grounded_edge_count must equal "
                "edge_ids length"
            )
        return self


class N11GroundedBridgeSearchResult(
    StrictModel
):
    schema_version: Literal[
        "n11-grounded-bridge-search-result-v1"
    ] = "n11-grounded-bridge-search-result-v1"

    search_id: str = Field(
        min_length=1
    )

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    status: Literal[
        "FOUND_GROUNDED_BRIDGE_CANDIDATES",
        "ABSTAIN_NO_DIRECT_SCIENTIFIC_CHAIN",
    ]

    reviewed_packet_ids: list[str]

    reviewed_path_count: int = Field(
        ge=0
    )

    direct_scientific_chain_count: int = Field(
        ge=0
    )

    rejected_path_class_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    rejection_reason_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    candidates: list[
        N11GroundedBridgeCandidate
    ]

    reason_codes: list[str] = Field(
        min_length=1
    )

    production_authority: Literal[
        False
    ] = False

    @model_validator(
        mode="after"
    )
    def _status_consistency(
        self,
    ) -> "N11GroundedBridgeSearchResult":
        if (
            self.status
            == "FOUND_GROUNDED_BRIDGE_CANDIDATES"
            and not self.candidates
        ):
            raise ValueError(
                "found status requires candidates"
            )

        if (
            self.status
            == "ABSTAIN_NO_DIRECT_SCIENTIFIC_CHAIN"
            and self.candidates
        ):
            raise ValueError(
                "abstain status cannot contain candidates"
            )

        if (
            self.direct_scientific_chain_count
            != len(self.candidates)
        ):
            raise ValueError(
                "direct_scientific_chain_count must equal "
                "candidate count"
            )

        return self


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _unique(
    values: list[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()
        if not text:
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def build_grounded_bridge_query_plan(
    opportunity: N11MissingBridgeOpportunity,
) -> N11GroundedBridgeQueryPlan:
    """Compile D1 search anchors into bounded graph queries.

    This function does not infer or assert a bridge relation.
    Every query uses only vocabulary already present in the
    D1 opportunity.
    """

    factor_terms = _unique(
        opportunity.factor_identity_terms
    )

    base_terms = _unique(
        opportunity.base_relation_terms
    )

    queries: list[
        N11GroundedBridgeQuery
    ] = []

    for factor in factor_terms:
        for base in base_terms:
            queries.append(
                N11GroundedBridgeQuery(
                    query_id=_stable_id(
                        "n11_bridge_query",
                        opportunity.opportunity_id,
                        factor,
                        base,
                    ),
                    source_missing_bridge_opportunity_id=(
                        opportunity.opportunity_id
                    ),
                    factor_identity_term=factor,
                    base_context_term=base,
                )
            )

    return N11GroundedBridgeQueryPlan(
        plan_id=_stable_id(
            "n11_bridge_query_plan",
            opportunity.opportunity_id,
            *[
                query.query_id
                for query in queries
            ],
        ),
        source_missing_bridge_opportunity_id=(
            opportunity.opportunity_id
        ),
        queries=queries,
    )


def _classify_path(
    *,
    packet: GraphExplorerPacket,
    path_index: int,
    semantics: DiscoverySemantics,
) -> tuple[
    Literal[
        "DIRECT_SCIENTIFIC_CHAIN",
        "COMMON_ANCHOR_CONTEXT",
        "NAVIGATION_ONLY",
    ],
    str | None,
]:
    path = packet.paths[path_index]

    if not path.steps:
        return (
            "NAVIGATION_ONLY",
            "empty_path",
        )

    # A factor -> base scientific chain must preserve the
    # scientific direction throughout. A reverse traversal is
    # exactly the pattern that can manufacture a common-anchor
    # pseudo-bridge:
    #
    # factor <- shared-context -> outcome
    if any(
        str(step.traversal_direction).lower()
        != "forward"
        for step in path.steps
    ):
        return (
            "COMMON_ANCHOR_CONTEXT",
            "reverse_scientific_direction",
        )

    if any(
        step.requires_verification
        for step in path.steps
    ):
        return (
            "NAVIGATION_ONLY",
            "candidate_or_unverified_edge",
        )

    for step in path.steps:
        edge = packet.evidence_catalog.edges.get(
            step.edge_evidence_ref
        )

        if edge is None:
            return (
                "NAVIGATION_ONLY",
                "missing_edge_evidence",
            )

        if (
            edge.provenance_status
            != "grounded_pointer"
        ):
            return (
                "NAVIGATION_ONLY",
                "edge_not_pointer_grounded",
            )

        if (
            is_alignment_edge(step)
            or is_alignment_edge(edge)
        ):
            return (
                "NAVIGATION_ONLY",
                "alignment_dependent",
            )

        # Structural/navigation scaffold edges may connect two
        # independently grounded facts without grounding the
        # factor -> base scientific relation itself.
        if (
            is_scaffold_edge(
                step,
                semantics,
            )
            or is_scaffold_edge(
                edge,
                semantics,
            )
        ):
            return (
                "COMMON_ANCHOR_CONTEXT",
                "scaffold_anchor_only",
            )

    return (
        "DIRECT_SCIENTIFIC_CHAIN",
        None,
    )


def evaluate_grounded_bridge_packets(
    *,
    opportunity: N11MissingBridgeOpportunity,
    query_plan: N11GroundedBridgeQueryPlan,
    packets: list[
        GraphExplorerPacket
    ],
    semantics: DiscoverySemantics,
) -> N11GroundedBridgeSearchResult:
    """Adjudicate graph retrieval without inventing relations.

    Only fully forward, pointer-grounded, non-alignment,
    non-scaffold scientific paths are eligible.
    """

    if (
        query_plan.source_missing_bridge_opportunity_id
        != opportunity.opportunity_id
    ):
        raise ValueError(
            "query plan does not belong to opportunity"
        )

    allowed_pairs = {
        (
            query.factor_identity_term,
            query.base_context_term,
        )
        for query in query_plan.queries
    }

    candidates: list[
        N11GroundedBridgeCandidate
    ] = []

    reviewed_packet_ids: list[str] = []
    reviewed_path_count = 0

    rejected_classes: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()

    seen_candidate_keys: set[
        tuple[str, ...]
    ] = set()

    for packet in packets:
        source_query = str(
            packet.task.source_query or ""
        ).strip()

        target_query = str(
            packet.task.target_query or ""
        ).strip()

        if (
            source_query,
            target_query,
        ) not in allowed_pairs:
            raise ValueError(
                "packet query pair not present in "
                "D2 query plan: "
                f"{source_query!r} -> "
                f"{target_query!r}"
            )

        reviewed_packet_ids.append(
            packet.packet_id
        )

        for path_index, path in enumerate(
            packet.paths
        ):
            reviewed_path_count += 1

            (
                path_class,
                rejection_reason,
            ) = _classify_path(
                packet=packet,
                path_index=path_index,
                semantics=semantics,
            )

            if (
                path_class
                != "DIRECT_SCIENTIFIC_CHAIN"
            ):
                rejected_classes[
                    path_class
                ] += 1

                if rejection_reason:
                    rejected_reasons[
                        rejection_reason
                    ] += 1

                continue

            edge_ids: list[str] = []
            relation_texts: list[str] = []
            paper_ids: list[str] = []

            for step in path.steps:
                edge = (
                    packet
                    .evidence_catalog
                    .edges[
                        step.edge_evidence_ref
                    ]
                )

                edge_ids.append(
                    edge.edge_id
                )

                relation_texts.append(
                    f"{edge.scientific_source} "
                    f"--{edge.relation}--> "
                    f"{edge.scientific_target}"
                )

                paper_ids.extend(
                    edge.source_paper_ids
                )

            paper_ids = _unique(
                paper_ids
            )

            # An edge may have a grounded pointer, but D2 needs
            # auditable literature provenance for operator
            # reconsideration.
            if not paper_ids:
                rejected_classes[
                    "NAVIGATION_ONLY"
                ] += 1
                rejected_reasons[
                    "no_source_paper_provenance"
                ] += 1
                continue

            candidate_key = tuple(
                [
                    path.endpoint.source_node_id,
                    *edge_ids,
                    path.endpoint.target_node_id,
                ]
            )

            if candidate_key in seen_candidate_keys:
                continue

            seen_candidate_keys.add(
                candidate_key
            )

            candidates.append(
                N11GroundedBridgeCandidate(
                    bridge_candidate_id=_stable_id(
                        "n11_grounded_bridge",
                        opportunity.opportunity_id,
                        packet.packet_id,
                        path.path_id,
                        *edge_ids,
                    ),
                    source_missing_bridge_opportunity_id=(
                        opportunity.opportunity_id
                    ),
                    factor_node_ids=[
                        path.endpoint.source_node_id
                    ],
                    intermediate_node_ids=list(
                        path.node_ids[1:-1]
                    ),
                    base_context_node_ids=[
                        path.endpoint.target_node_id
                    ],
                    packet_ids=[
                        packet.packet_id
                    ],
                    path_ids=[
                        path.path_id
                    ],
                    edge_ids=edge_ids,
                    source_paper_ids=paper_ids,
                    scientific_relation_texts=(
                        relation_texts
                    ),
                    grounded_edge_count=len(
                        edge_ids
                    ),
                )
            )

    search_id = _stable_id(
        "n11_grounded_bridge_search",
        opportunity.opportunity_id,
        query_plan.plan_id,
        *sorted(reviewed_packet_ids),
    )

    if candidates:
        return N11GroundedBridgeSearchResult(
            search_id=search_id,
            source_missing_bridge_opportunity_id=(
                opportunity.opportunity_id
            ),
            status=(
                "FOUND_GROUNDED_BRIDGE_CANDIDATES"
            ),
            reviewed_packet_ids=_unique(
                reviewed_packet_ids
            ),
            reviewed_path_count=(
                reviewed_path_count
            ),
            direct_scientific_chain_count=len(
                candidates
            ),
            rejected_path_class_counts=dict(
                rejected_classes
            ),
            rejection_reason_counts=dict(
                rejected_reasons
            ),
            candidates=candidates,
            reason_codes=[
                "direct_scientific_chain_found"
            ],
        )

    return N11GroundedBridgeSearchResult(
        search_id=search_id,
        source_missing_bridge_opportunity_id=(
            opportunity.opportunity_id
        ),
        status=(
            "ABSTAIN_NO_DIRECT_SCIENTIFIC_CHAIN"
        ),
        reviewed_packet_ids=_unique(
            reviewed_packet_ids
        ),
        reviewed_path_count=(
            reviewed_path_count
        ),
        direct_scientific_chain_count=0,
        rejected_path_class_counts=dict(
            rejected_classes
        ),
        rejection_reason_counts=dict(
            rejected_reasons
        ),
        candidates=[],
        reason_codes=[
            "no_direct_scientific_chain_found"
        ],
    )
