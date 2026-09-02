from __future__ import annotations

import pytest

from pipeline_core.discovery.explorer_contracts import (
    GraphExplorerPacket,
)
from pipeline_core.discovery.nonobviousness_grounded_bridge import (
    build_grounded_bridge_query_plan,
    evaluate_grounded_bridge_packets,
)
from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeOpportunity,
)
from pipeline_core.domain.domain_profile import (
    DiscoverySemantics,
)


def semantics() -> DiscoverySemantics:
    return DiscoverySemantics(
        generic_entity_types=frozenset(),
        mechanism_node_markers=(),
        mechanism_relation_markers=(
            "MODULAT",
            "AFFECT",
        ),
        scaffold_relations=frozenset({
            "HAS_COMPONENT",
            "HAS_ARCHITECTURE",
        }),
        context_node_types=frozenset(),
    )


def opportunity() -> N11MissingBridgeOpportunity:
    return N11MissingBridgeOpportunity(
        opportunity_id="n11_missing_bridge:test",
        source_portfolio_id="portfolio:test",
        source_hypothesis_id="hypothesis:test",
        source_claim_id="claim:test",
        source_execution_plan_id="plan:test",
        factor_identity_terms=[
            "interparticle spacing",
        ],
        base_relation_terms=[
            "SERS response",
            "electromagnetic enhancement",
        ],
        bridge_target_text_for_audit=(
            "interparticle spacing may alter "
            "relative mechanistic contribution"
        ),
        full_relation_text_for_audit=(
            "interparticle spacing changes "
            "measured SERS behavior"
        ),
        bridge_retrieval_terms_for_audit=[
            "interparticle spacing",
            "SERS response",
        ],
        established_base_work_ids=[
            "work:base",
        ],
        established_factor_work_ids=[
            "work:factor",
        ],
    )


def packet(
    *,
    relation: str = "MODULATES",
    traversal_direction: str = "forward",
    edge_class: str = "",
    graph_layer: str = "",
    provenance_status: str = "grounded_pointer",
    evidence_pointer_source: str = "edge_sidecar",
    source_paper_ids: list[str] | None = None,
    requires_verification: bool = False,
    source_query: str = "interparticle spacing",
    target_query: str = "SERS response",
) -> GraphExplorerPacket:
    papers = (
        ["paper:test"]
        if source_paper_ids is None
        else source_paper_ids
    )

    return GraphExplorerPacket.model_validate({
        "domain_profile_id": "sers_au_ag",
        "packet_id": (
            "packet:"
            + relation
            + ":"
            + traversal_direction
            + ":"
            + provenance_status
        ),
        "packet_sha256": "sha:test",
        "task": {
            "task_id": "task:test",
            "question": "test question",
            "source_query": source_query,
            "target_query": target_query,
            "traversal_mode": "mechanism",
        },
        "corpus": {
            "corpus_id": "sers500_final_v2",
            "projection_mode": "mechanism",
            "papers": [],
            "substrate_version": "test",
        },
        "retrieval_summary": {
            "algorithm": "top_n",
            "returned_path_count": 1,
        },
        "direct_concept_hits": [],
        "paths": [
            {
                "path_id": "path:test",
                "bundle_rank": 1,
                "endpoint": {
                    "source_node_id": "node:factor",
                    "target_node_id": "node:base",
                },
                "node_ids": [
                    "node:factor",
                    "node:base",
                ],
                "steps": [
                    {
                        "navigation_source": "node:factor",
                        "navigation_target": "node:base",
                        "traversal_direction": (
                            traversal_direction
                        ),
                        "scientific_source": "node:factor",
                        "relation": relation,
                        "scientific_target": "node:base",
                        "selected_original_edge_id": (
                            "edge:test"
                        ),
                        "edge_evidence_ref": "edge:test",
                        "edge_class": edge_class,
                        "requires_verification": (
                            requires_verification
                        ),
                    }
                ],
                "visited_paper_ids": papers,
                "supporting_paper_ids": papers,
                "quality": {},
            }
        ],
        "evidence_catalog": {
            "nodes": {},
            "edges": {
                "edge:test": {
                    "edge_id": "edge:test",
                    "scientific_source": "node:factor",
                    "relation": relation,
                    "scientific_target": "node:base",
                    "graph_layer": graph_layer,
                    "requires_verification": (
                        requires_verification
                    ),
                    "source_paper_ids": papers,
                    "evidence_pointers": (
                        [{"pointer": "p1"}]
                        if provenance_status
                        == "grounded_pointer"
                        else []
                    ),
                    "evidence_pointer_source": (
                        evidence_pointer_source
                    ),
                    "provenance_status": (
                        provenance_status
                    ),
                }
            },
        },
        "alignment_contexts": [],
        "provenance_summary": {
            "strict_provenance": True,
            "edge_count": 1,
            "pointer_grounded_edge_count": (
                1
                if provenance_status
                == "grounded_pointer"
                else 0
            ),
            "pointer_recovered_from_traversal_count": 0,
            "derived_alignment_edge_count": (
                1
                if graph_layer
                == "corpus_alignment"
                else 0
            ),
            "missing_pointer_edge_count": (
                0
                if provenance_status
                == "grounded_pointer"
                else 1
            ),
            "materialized_node_count": 0,
            "suppressed_alignment_member_node_count": 0,
        },
    })


def run(
    candidate_packet: GraphExplorerPacket,
):
    opp = opportunity()
    plan = build_grounded_bridge_query_plan(
        opp
    )

    return evaluate_grounded_bridge_packets(
        opportunity=opp,
        query_plan=plan,
        packets=[candidate_packet],
        semantics=semantics(),
    )


def test_query_plan_is_source_bounded():
    opp = opportunity()

    plan = build_grounded_bridge_query_plan(
        opp
    )

    assert len(plan.queries) == 2

    assert {
        (
            row.factor_identity_term,
            row.base_context_term,
        )
        for row in plan.queries
    } == {
        (
            "interparticle spacing",
            "SERS response",
        ),
        (
            "interparticle spacing",
            "electromagnetic enhancement",
        ),
    }

    assert all(
        row.max_alignment_edges == 0
        for row in plan.queries
    )

    assert all(
        row.production_authority is False
        for row in plan.queries
    )


def test_forward_grounded_scientific_chain_is_eligible():
    result = run(
        packet()
    )

    assert (
        result.status
        == "FOUND_GROUNDED_BRIDGE_CANDIDATES"
    )

    assert (
        result.direct_scientific_chain_count
        == 1
    )

    candidate = result.candidates[0]

    assert (
        candidate.path_class
        == "DIRECT_SCIENTIFIC_CHAIN"
    )

    assert candidate.edge_ids == [
        "edge:test"
    ]

    assert candidate.source_paper_ids == [
        "paper:test"
    ]

    assert (
        candidate.eligible_for_operator_reconsideration
        is True
    )

    assert (
        candidate.production_authority
        is False
    )


def test_reverse_path_is_common_anchor_not_bridge():
    result = run(
        packet(
            traversal_direction="reverse",
        )
    )

    assert (
        result.status
        == "ABSTAIN_NO_DIRECT_SCIENTIFIC_CHAIN"
    )

    assert (
        result.rejected_path_class_counts[
            "COMMON_ANCHOR_CONTEXT"
        ]
        == 1
    )

    assert (
        result.rejection_reason_counts[
            "reverse_scientific_direction"
        ]
        == 1
    )


def test_scaffold_path_is_common_anchor_not_bridge():
    result = run(
        packet(
            relation="HAS_COMPONENT",
        )
    )

    assert (
        result.status
        == "ABSTAIN_NO_DIRECT_SCIENTIFIC_CHAIN"
    )

    assert (
        result.rejection_reason_counts[
            "scaffold_anchor_only"
        ]
        == 1
    )


def test_alignment_path_is_navigation_only():
    result = run(
        packet(
            edge_class="registry_alignment",
            graph_layer="corpus_alignment",
        )
    )

    assert (
        result.status
        == "ABSTAIN_NO_DIRECT_SCIENTIFIC_CHAIN"
    )

    assert (
        result.rejected_path_class_counts[
            "NAVIGATION_ONLY"
        ]
        == 1
    )

    assert (
        result.rejection_reason_counts[
            "alignment_dependent"
        ]
        == 1
    )


def test_ungrounded_edge_is_navigation_only():
    result = run(
        packet(
            provenance_status="missing_pointer",
            evidence_pointer_source="missing",
        )
    )

    assert (
        result.status
        == "ABSTAIN_NO_DIRECT_SCIENTIFIC_CHAIN"
    )

    assert (
        result.rejection_reason_counts[
            "edge_not_pointer_grounded"
        ]
        == 1
    )


def test_missing_paper_provenance_abstains():
    result = run(
        packet(
            source_paper_ids=[],
        )
    )

    assert (
        result.status
        == "ABSTAIN_NO_DIRECT_SCIENTIFIC_CHAIN"
    )

    assert (
        result.rejection_reason_counts[
            "no_source_paper_provenance"
        ]
        == 1
    )


def test_packet_must_belong_to_query_plan():
    opp = opportunity()

    plan = build_grounded_bridge_query_plan(
        opp
    )

    bad_packet = packet(
        target_query="unplanned target",
    )

    with pytest.raises(
        ValueError,
        match="packet query pair not present",
    ):
        evaluate_grounded_bridge_packets(
            opportunity=opp,
            query_plan=plan,
            packets=[bad_packet],
            semantics=semantics(),
        )
