from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.scientific_claim_centrality import (
    ScientificClaimCentralityRecord,
    ScientificClaimCentralityReport,
    ScientificHypothesisCentralitySummary,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphClaimSummary,
    ScientificClaimEvidenceGraphHypothesisSummary,
    ScientificClaimEvidenceGraphNode,
    ScientificClaimEvidenceGraphEdge,
    ScientificClaimEvidenceGraphReport,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    build_scientific_hypothesis_evidence_aggregation,
)


def _hashed_graph(*, claims, hypotheses, nodes, edges):
    body = {
        "schema_version":
            "scientific-claim-evidence-graph-report-v1",
        "source_relation_ir_report_id":
            "relation:1",
        "source_projection_report_id":
            "projection:1",
        "source_candidate_report_id":
            "candidate:1",
        "source_adjudication_report_id":
            "adjudication:1",
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
            for row in claims
        ],
        "hypothesis_summaries": [
            row.model_dump(mode="json")
            for row in hypotheses
        ],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "hypothesis_count": len(hypotheses),
        "claim_count": len(claims),
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
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ScientificClaimEvidenceGraphReport(
        **body,
        graph_id=(
            "scientific_claim_evidence_graph:"
            + digest[:20]
        ),
        graph_sha256=digest,
    )


def _claim(
    *,
    h,
    c,
    role="NOVELTY_BEARING",
    presented=2,
    classified=2,
    relationship_counts=None,
):
    return ScientificClaimEvidenceGraphClaimSummary(
        hypothesis_id=h,
        claim_id=c,
        relation_ir_id="relation:" + c,
        claim_kind="moderator_interaction",
        novelty_selection_role=role,
        typing_status="READY",
        adjudication_status="REVIEWED",
        relation_state="COMPONENTS_OR_CONTEXT_ONLY",
        concept_node_ids=["concept:" + c],
        projection_node_ids=["projection:" + c],
        presented_work_node_ids=[
            f"work-node:{c}:{i}"
            for i in range(presented)
        ],
        classified_work_node_ids=[
            f"work-node:{c}:{i}"
            for i in range(classified)
        ],
        unclassified_work_node_ids=[
            f"work-node:{c}:{i}"
            for i in range(classified, presented)
        ],
        presented_work_count=presented,
        classified_work_count=classified,
        unclassified_work_count=presented - classified,
        relationship_counts=relationship_counts or {},
    )


def _hyp(h, claim_ids):
    return ScientificClaimEvidenceGraphHypothesisSummary(
        hypothesis_id=h,
        claim_ids=claim_ids,
        novelty_bearing_claim_ids=[
            claim_id
            for claim_id in claim_ids
            if claim_id.startswith("n")
        ],
        reviewed_claim_ids=claim_ids,
        unreviewed_claim_ids=[],
        claim_count=len(claim_ids),
        reviewed_claim_count=len(claim_ids),
        unreviewed_claim_count=0,
    )


def _centrality_report(graph, records, summaries):
    body = {
        "schema_version":
            "scientific-claim-centrality-report-v1",
        "source_claim_evidence_graph_id":
            graph.graph_id,
        "claim_records": [
            row.model_dump(mode="json")
            for row in records
        ],
        "pair_topologies": [],
        "hypothesis_summaries": [
            row.model_dump(mode="json")
            for row in summaries
        ],
        "hypothesis_count": len(summaries),
        "claim_count": len(records),
        "pair_topology_count": 0,
        "single_claim_hypothesis_count": sum(
            row.topology_state
            == "SINGLE_CLAIM_DEGENERATE"
            for row in summaries
        ),
        "multi_claim_connected_hypothesis_count": sum(
            row.topology_state
            == "MULTI_CLAIM_CONNECTED"
            for row in summaries
        ),
        "multi_claim_no_shared_concept_hypothesis_count": sum(
            row.topology_state
            == "MULTI_CLAIM_NO_EXACT_SHARED_CONCEPTS"
            for row in summaries
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
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ScientificClaimCentralityReport(
        **body,
        report_id=(
            "scientific_claim_centrality:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


def _centrality(
    *,
    h,
    c,
    role,
    rank,
    score,
    pressure,
    classified_fraction=1.0,
    topology="MULTI_CLAIM_CONNECTED",
):
    return ScientificClaimCentralityRecord(
        hypothesis_id=h,
        claim_id=c,
        novelty_selection_role=role,
        role_precedence={
            "NOVELTY_BEARING": 0,
            "REQUIRED_ENABLING_RELATION": 1,
            "TESTING_PREDICTION": 2,
            "AUXILIARY": 3,
        }[role],
        claim_kind="moderator_interaction",
        typing_status="READY",
        topology_state=topology,
        hypothesis_claim_count=1 if topology == "SINGLE_CLAIM_DEGENERATE" else 2,
        concept_node_count=1,
        neighbor_claim_count=0 if topology == "SINGLE_CLAIM_DEGENERATE" else 1,
        neighbor_degree_centrality=1.0,
        shared_concept_node_ids=["shared"] if topology != "SINGLE_CLAIM_DEGENERATE" else ["only"],
        shared_concept_count=1,
        shared_concept_fraction=1.0,
        mean_pairwise_concept_jaccard=score,
        structural_centrality_score=score,
        centrality_rank_within_hypothesis=rank,
        centrality_rank_tied=False,
        classification_coverage_state=(
            "FULLY_CLASSIFIED"
            if classified_fraction == 1.0
            else "PARTIALLY_CLASSIFIED"
        ),
        classification_fraction=classified_fraction,
        evidence_pressure_state=pressure,
        strong_evidence_pressure_present=(
            pressure
            in {
                "DIRECT_PRIOR_ART_PRESSURE",
                "LOWER_ORDER_PRIOR_ART_PRESSURE",
                "COUNTEREVIDENCE_PRESSURE",
                "CONFLICTING_EVIDENCE_PRESSURE",
                "MIXED_STRONG_SIGNAL_PRESSURE",
            }
        ),
    )


def _work_node(work_id):
    return ScientificClaimEvidenceGraphNode(
        node_id="node:" + work_id,
        node_kind="WORK",
        label=work_id,
        source_ids=[work_id],
        source_work_ids=[work_id],
        title=work_id,
    )


def _adjudication_edge(
    *,
    h,
    c,
    work_id,
    relationship,
):
    return ScientificClaimEvidenceGraphEdge(
        edge_id=f"edge:{c}:{work_id}:{relationship}",
        edge_kind="WORK_ADJUDICATED_TO_CLAIM",
        source_node_id="node:" + work_id,
        target_node_id="claim-node:" + c,
        hypothesis_id=h,
        claim_id=c,
        review_status="CLASSIFIED",
        relationship=relationship,
        original_relationship=relationship,
        confidence=0.9,
        evidence_span=(
            "exact evidence span"
            if relationship
            in {
                "DIRECT_PRIOR_ART",
                "PARTIAL_PRIOR_ART",
                "LOWER_ORDER_RELATION_PRIOR_ART",
                "DIRECTIONAL_COUNTEREVIDENCE",
                "CONTEXTUAL_CONFLICT",
                "CONFLICTING_PRIOR_ART",
            }
            else ""
        ),
        rationale="test",
    )


def _claim_node(h, c):
    return ScientificClaimEvidenceGraphNode(
        node_id="claim-node:" + c,
        node_kind="CLAIM",
        label=c,
        hypothesis_ids=[h],
        claim_ids=[c],
        source_ids=[c],
        relation_ir_id="relation:" + c,
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        typing_status="READY",
    )


def _summary(
    *,
    h,
    claim_ids,
    novelty_ids,
    topology,
    fully_classified=True,
):
    return ScientificHypothesisCentralitySummary(
        hypothesis_id=h,
        topology_state=topology,
        claim_ids=claim_ids,
        novelty_bearing_claim_ids=novelty_ids,
        role_ordered_claim_ids=claim_ids,
        centrality_ordered_claim_ids=claim_ids,
        claim_count=len(claim_ids),
        exact_shared_claim_pair_count=(
            0 if len(claim_ids) == 1 else 1
        ),
        total_claim_pair_count=(
            len(claim_ids) * (len(claim_ids) - 1) // 2
        ),
        classification_coverage_states={
            (
                "FULLY_CLASSIFIED"
                if fully_classified
                else "PARTIALLY_CLASSIFIED"
            ): len(claim_ids)
        },
        evidence_pressure_states={},
        strong_pressure_claim_ids=[],
        all_claims_fully_classified=fully_classified,
        any_unclassified_presented_work=not fully_classified,
        centrality_is_degenerate=(
            topology == "SINGLE_CLAIM_DEGENERATE"
        ),
        aggregation_readiness=(
            "STRUCTURAL_CENTRALITY_READY_FULL_CLASSIFICATION"
            if fully_classified
            else "STRUCTURAL_CENTRALITY_READY_EPISTEMICALLY_PARTIAL"
        ),
    )


def test_component_only_single_claim_produces_no_strong_pressure_profile():
    h, c = "h1", "n1"
    claim = _claim(
        h=h,
        c=c,
        relationship_counts={"COMPONENT_ONLY": 1},
        presented=2,
        classified=1,
    )
    nodes = [
        _claim_node(h, c),
        _work_node("w1"),
    ]
    edges = [
        _adjudication_edge(
            h=h,
            c=c,
            work_id="w1",
            relationship="COMPONENT_ONLY",
        )
    ]
    graph = _hashed_graph(
        claims=[claim],
        hypotheses=[_hyp(h, [c])],
        nodes=nodes,
        edges=edges,
    )
    cent = _centrality(
        h=h,
        c=c,
        role="NOVELTY_BEARING",
        rank=1,
        score=1.0,
        pressure="COMPONENT_CONTEXT_ONLY",
        classified_fraction=0.5,
        topology="SINGLE_CLAIM_DEGENERATE",
    )
    centrality = _centrality_report(
        graph,
        [cent],
        [
            _summary(
                h=h,
                claim_ids=[c],
                novelty_ids=[c],
                topology="SINGLE_CLAIM_DEGENERATE",
                fully_classified=False,
            )
        ],
    )

    report = build_scientific_hypothesis_evidence_aggregation(
        graph=graph,
        centrality=centrality,
    )

    agg = report.hypothesis_aggregations[0]
    assert agg.evidence_impact_profile == (
        "NO_STRONG_RELATION_PRESSURE"
    )
    assert agg.weak_signal_profile == (
        "COMPONENT_CONTEXT_ONLY"
    )
    assert agg.epistemic_coverage_profile == (
        "PARTIAL_CLASSIFICATION_COVERAGE"
    )
    assert report.novelty_verdict_created is False


def test_strong_pressure_on_novelty_bearing_highest_centrality_claim_is_flagged():
    h = "h1"
    claims = [
        _claim(
            h=h,
            c="n1",
            role="NOVELTY_BEARING",
            relationship_counts={"DIRECT_PRIOR_ART": 1},
        ),
        _claim(
            h=h,
            c="a1",
            role="AUXILIARY",
            relationship_counts={},
        ),
    ]
    nodes = [
        _claim_node(h, "n1"),
        _claim_node(h, "a1"),
        _work_node("w1"),
    ]
    edges = [
        _adjudication_edge(
            h=h,
            c="n1",
            work_id="w1",
            relationship="DIRECT_PRIOR_ART",
        )
    ]
    graph = _hashed_graph(
        claims=claims,
        hypotheses=[_hyp(h, ["n1", "a1"])],
        nodes=nodes,
        edges=edges,
    )
    records = [
        _centrality(
            h=h,
            c="n1",
            role="NOVELTY_BEARING",
            rank=1,
            score=0.8,
            pressure="DIRECT_PRIOR_ART_PRESSURE",
        ),
        _centrality(
            h=h,
            c="a1",
            role="AUXILIARY",
            rank=2,
            score=0.2,
            pressure="NO_MATERIAL_SIGNAL_IN_CLASSIFIED_SUBSET",
        ),
    ]
    centrality = _centrality_report(
        graph,
        records,
        [
            _summary(
                h=h,
                claim_ids=["n1", "a1"],
                novelty_ids=["n1"],
                topology="MULTI_CLAIM_CONNECTED",
            )
        ],
    )

    report = build_scientific_hypothesis_evidence_aggregation(
        graph=graph,
        centrality=centrality,
    )

    agg = report.hypothesis_aggregations[0]
    assert agg.evidence_impact_profile == (
        "NOVELTY_BEARING_HIGHEST_CENTRALITY_STRONG_PRESSURE"
    )
    assert agg.strong_pressure_claim_ids == ["n1"]
    assert agg.novelty_bearing_highest_centrality_strong_pressure_claim_ids == [
        "n1"
    ]


def test_strong_pressure_on_auxiliary_highest_centrality_claim_is_not_promoted_to_novelty_bearing():
    h = "h1"
    claims = [
        _claim(
            h=h,
            c="n1",
            role="NOVELTY_BEARING",
            relationship_counts={},
        ),
        _claim(
            h=h,
            c="a1",
            role="AUXILIARY",
            relationship_counts={
                "LOWER_ORDER_RELATION_PRIOR_ART": 1
            },
        ),
    ]
    nodes = [
        _claim_node(h, "n1"),
        _claim_node(h, "a1"),
        _work_node("w1"),
    ]
    edges = [
        _adjudication_edge(
            h=h,
            c="a1",
            work_id="w1",
            relationship="LOWER_ORDER_RELATION_PRIOR_ART",
        )
    ]
    graph = _hashed_graph(
        claims=claims,
        hypotheses=[_hyp(h, ["n1", "a1"])],
        nodes=nodes,
        edges=edges,
    )
    records = [
        _centrality(
            h=h,
            c="n1",
            role="NOVELTY_BEARING",
            rank=2,
            score=0.2,
            pressure="NO_MATERIAL_SIGNAL_IN_CLASSIFIED_SUBSET",
        ),
        _centrality(
            h=h,
            c="a1",
            role="AUXILIARY",
            rank=1,
            score=0.8,
            pressure="LOWER_ORDER_PRIOR_ART_PRESSURE",
        ),
    ]
    centrality = _centrality_report(
        graph,
        records,
        [
            _summary(
                h=h,
                claim_ids=["n1", "a1"],
                novelty_ids=["n1"],
                topology="MULTI_CLAIM_CONNECTED",
            )
        ],
    )

    report = build_scientific_hypothesis_evidence_aggregation(
        graph=graph,
        centrality=centrality,
    )
    agg = report.hypothesis_aggregations[0]

    assert agg.evidence_impact_profile == (
        "HIGHEST_CENTRALITY_STRONG_PRESSURE_OUTSIDE_NOVELTY_BEARING"
    )
    assert agg.novelty_bearing_strong_pressure_claim_ids == []


def test_novelty_bearing_strong_pressure_outside_highest_centrality_is_distinguished():
    h = "h1"
    claims = [
        _claim(
            h=h,
            c="n1",
            relationship_counts={
                "DIRECTIONAL_COUNTEREVIDENCE": 1
            },
        ),
        _claim(
            h=h,
            c="n2",
            relationship_counts={},
        ),
    ]
    nodes = [
        _claim_node(h, "n1"),
        _claim_node(h, "n2"),
        _work_node("w1"),
    ]
    edges = [
        _adjudication_edge(
            h=h,
            c="n1",
            work_id="w1",
            relationship="DIRECTIONAL_COUNTEREVIDENCE",
        )
    ]
    graph = _hashed_graph(
        claims=claims,
        hypotheses=[_hyp(h, ["n1", "n2"])],
        nodes=nodes,
        edges=edges,
    )
    records = [
        _centrality(
            h=h,
            c="n1",
            role="NOVELTY_BEARING",
            rank=2,
            score=0.2,
            pressure="COUNTEREVIDENCE_PRESSURE",
        ),
        _centrality(
            h=h,
            c="n2",
            role="NOVELTY_BEARING",
            rank=1,
            score=0.8,
            pressure="NO_MATERIAL_SIGNAL_IN_CLASSIFIED_SUBSET",
        ),
    ]
    centrality = _centrality_report(
        graph,
        records,
        [
            _summary(
                h=h,
                claim_ids=["n1", "n2"],
                novelty_ids=["n1", "n2"],
                topology="MULTI_CLAIM_CONNECTED",
            )
        ],
    )

    report = build_scientific_hypothesis_evidence_aggregation(
        graph=graph,
        centrality=centrality,
    )
    agg = report.hypothesis_aggregations[0]
    assert agg.evidence_impact_profile == (
        "NOVELTY_BEARING_STRONG_PRESSURE_OUTSIDE_HIGHEST_CENTRALITY"
    )


def test_review_priority_is_role_then_centrality_then_pressure_without_weighted_score():
    h = "h1"
    claims = [
        _claim(
            h=h,
            c="n1",
            role="NOVELTY_BEARING",
        ),
        _claim(
            h=h,
            c="a1",
            role="AUXILIARY",
            relationship_counts={"DIRECT_PRIOR_ART": 1},
        ),
    ]
    nodes = [
        _claim_node(h, "n1"),
        _claim_node(h, "a1"),
        _work_node("w1"),
    ]
    edges = [
        _adjudication_edge(
            h=h,
            c="a1",
            work_id="w1",
            relationship="DIRECT_PRIOR_ART",
        )
    ]
    graph = _hashed_graph(
        claims=claims,
        hypotheses=[_hyp(h, ["n1", "a1"])],
        nodes=nodes,
        edges=edges,
    )
    records = [
        _centrality(
            h=h,
            c="n1",
            role="NOVELTY_BEARING",
            rank=2,
            score=0.2,
            pressure="NO_MATERIAL_SIGNAL_IN_CLASSIFIED_SUBSET",
        ),
        _centrality(
            h=h,
            c="a1",
            role="AUXILIARY",
            rank=1,
            score=0.9,
            pressure="DIRECT_PRIOR_ART_PRESSURE",
        ),
    ]
    centrality = _centrality_report(
        graph,
        records,
        [
            _summary(
                h=h,
                claim_ids=["n1", "a1"],
                novelty_ids=["n1"],
                topology="MULTI_CLAIM_CONNECTED",
            )
        ],
    )

    report = build_scientific_hypothesis_evidence_aggregation(
        graph=graph,
        centrality=centrality,
    )
    agg = report.hypothesis_aggregations[0]

    # NOVELTY_BEARING remains first in the review queue even though the
    # auxiliary claim is structurally more central and carries direct prior art.
    assert agg.review_priority_claim_ids == ["n1", "a1"]
    assert (
        report.role_precedence_is_lexicographic_not_numeric_weight
        is True
    )
    assert report.novelty_verdict_created is False
