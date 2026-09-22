from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.positive_nonobviousness_basis import (
    build_positive_nonobviousness_basis_report,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphClaimSummary,
    ScientificClaimEvidenceGraphEdge,
    ScientificClaimEvidenceGraphHypothesisSummary,
    ScientificClaimEvidenceGraphNode,
    ScientificClaimEvidenceGraphReport,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    ScientificClaimEvidenceImpactRecord,
    ScientificHypothesisEvidenceAggregation,
    ScientificHypothesisEvidenceAggregationReport,
)


def _hash_payload(body):
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _graph(
    *,
    claim,
    hypothesis,
    nodes,
    edges,
):
    body = {
        "schema_version":
            "scientific-claim-evidence-graph-report-v1",
        "source_relation_ir_report_id": "relation:1",
        "source_projection_report_id": "projection:1",
        "source_candidate_report_id": "candidate:1",
        "source_adjudication_report_id": "adjudication:1",
        "nodes": [
            row.model_dump(mode="json")
            for row in nodes
        ],
        "edges": [
            row.model_dump(mode="json")
            for row in edges
        ],
        "claim_summaries": [
            claim.model_dump(mode="json")
        ],
        "hypothesis_summaries": [
            hypothesis.model_dump(mode="json")
        ],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "hypothesis_count": 1,
        "claim_count": 1,
        "concept_node_count": 0,
        "projection_node_count": 0,
        "work_node_count": sum(
            row.node_kind == "WORK"
            for row in nodes
        ),
        "presented_work_edge_count": 0,
        "adjudicated_relation_edge_count": sum(
            row.edge_kind
            == "WORK_ADJUDICATED_TO_CLAIM"
            for row in edges
        ),
        "unclassified_presentation_edge_count": 0,
        "graph_semantics":
            "deterministic_claim_evidence_topology_v1",
        "exact_normalized_concept_coalescing_only": True,
        "classified_and_unclassified_review_separated": True,
        "unclassified_is_not_negative_evidence": True,
        "absence_based_novelty_authorized": False,
        "positive_nonobviousness_authority_created": False,
        "centrality_scoring_performed": False,
        "aggregation_performed": False,
        "diagnostic_only": True,
        "n9_contract_changed": False,
        "n10_contract_changed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _hash_payload(body)
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
    claim_id="n1",
    role="NOVELTY_BEARING",
    relationship_counts=None,
    presented=1,
    classified=1,
):
    return ScientificClaimEvidenceGraphClaimSummary(
        hypothesis_id="h1",
        claim_id=claim_id,
        relation_ir_id="relation:" + claim_id,
        claim_kind="moderator_interaction",
        novelty_selection_role=role,
        typing_status="READY",
        adjudication_status="REVIEWED",
        relation_state="COMPONENTS_OR_CONTEXT_ONLY",
        concept_node_ids=["concept:" + claim_id],
        projection_node_ids=["projection:" + claim_id],
        presented_work_node_ids=[
            f"work-node:{claim_id}:{i}"
            for i in range(presented)
        ],
        classified_work_node_ids=[
            f"work-node:{claim_id}:{i}"
            for i in range(classified)
        ],
        unclassified_work_node_ids=[
            f"work-node:{claim_id}:{i}"
            for i in range(classified, presented)
        ],
        presented_work_count=presented,
        classified_work_count=classified,
        unclassified_work_count=presented - classified,
        relationship_counts=relationship_counts or {},
    )


def _hyp(claim_id="n1"):
    return ScientificClaimEvidenceGraphHypothesisSummary(
        hypothesis_id="h1",
        claim_ids=[claim_id],
        novelty_bearing_claim_ids=[claim_id],
        reviewed_claim_ids=[claim_id],
        unreviewed_claim_ids=[],
        claim_count=1,
        reviewed_claim_count=1,
        unreviewed_claim_count=0,
    )


def _work(work_id="w1"):
    return ScientificClaimEvidenceGraphNode(
        node_id="work-node:" + work_id,
        node_kind="WORK",
        label=work_id,
        source_ids=[work_id],
        source_work_ids=[work_id],
        title=work_id,
    )


def _claim_node(claim_id="n1", role="NOVELTY_BEARING"):
    return ScientificClaimEvidenceGraphNode(
        node_id="claim-node:" + claim_id,
        node_kind="CLAIM",
        label=claim_id,
        hypothesis_ids=["h1"],
        claim_ids=[claim_id],
        source_ids=[claim_id],
        relation_ir_id="relation:" + claim_id,
        claim_kind="moderator_interaction",
        novelty_selection_role=role,
        typing_status="READY",
    )


def _edge(
    *,
    relationship,
    work_id="w1",
    claim_id="n1",
):
    return ScientificClaimEvidenceGraphEdge(
        edge_id=(
            f"edge:{claim_id}:{work_id}:{relationship}"
        ),
        edge_kind="WORK_ADJUDICATED_TO_CLAIM",
        source_node_id="work-node:" + work_id,
        target_node_id="claim-node:" + claim_id,
        hypothesis_id="h1",
        claim_id=claim_id,
        review_status="CLASSIFIED",
        relationship=relationship,
        original_relationship=relationship,
        confidence=0.9,
        evidence_span=(
            "The reported relation was decoupled under matched conditions."
            if relationship
            in {
                "DIRECTIONAL_COUNTEREVIDENCE",
                "CONTEXTUAL_CONFLICT",
                "CONFLICTING_PRIOR_ART",
                "DIRECT_PRIOR_ART",
                "LOWER_ORDER_RELATION_PRIOR_ART",
                "PARTIAL_PRIOR_ART",
            }
            else ""
        ),
        rationale="test",
    )


def _impact(
    *,
    claim_id="n1",
    role="NOVELTY_BEARING",
    relationship=None,
    classified_fraction=1.0,
):
    mapping = {
        "DIRECT_PRIOR_ART":
            ("DIRECT_PRIOR_ART_PRESSURE", ["w1"], []),
        "LOWER_ORDER_RELATION_PRIOR_ART":
            ("LOWER_ORDER_PRIOR_ART_PRESSURE", ["w1"], []),
        "DIRECTIONAL_COUNTEREVIDENCE":
            ("COUNTEREVIDENCE_PRESSURE", ["w1"], []),
        "CONTEXTUAL_CONFLICT":
            ("COUNTEREVIDENCE_PRESSURE", ["w1"], []),
        "CONFLICTING_PRIOR_ART":
            ("CONFLICTING_EVIDENCE_PRESSURE", ["w1"], []),
        "COMPONENT_ONLY":
            ("COMPONENT_CONTEXT_ONLY", [], ["w1"]),
        None:
            ("NO_MATERIAL_SIGNAL_IN_CLASSIFIED_SUBSET", [], []),
    }
    pressure, strong, weak = mapping[relationship]

    kwargs = {
        "direct_prior_art_work_ids": [],
        "lower_order_prior_art_work_ids": [],
        "directional_counterevidence_work_ids": [],
        "contextual_conflict_work_ids": [],
        "conflicting_prior_art_work_ids": [],
        "partial_prior_art_work_ids": [],
        "component_only_work_ids": [],
    }
    strong_kinds = []
    weak_kinds = []

    if relationship == "DIRECT_PRIOR_ART":
        kwargs["direct_prior_art_work_ids"] = ["w1"]
        strong_kinds = ["DIRECT_PRIOR_ART"]
    elif relationship == "LOWER_ORDER_RELATION_PRIOR_ART":
        kwargs["lower_order_prior_art_work_ids"] = ["w1"]
        strong_kinds = ["LOWER_ORDER_RELATION_PRIOR_ART"]
    elif relationship == "DIRECTIONAL_COUNTEREVIDENCE":
        kwargs["directional_counterevidence_work_ids"] = ["w1"]
        strong_kinds = ["DIRECTIONAL_COUNTEREVIDENCE"]
    elif relationship == "CONTEXTUAL_CONFLICT":
        kwargs["contextual_conflict_work_ids"] = ["w1"]
        strong_kinds = ["CONTEXTUAL_CONFLICT"]
    elif relationship == "CONFLICTING_PRIOR_ART":
        kwargs["conflicting_prior_art_work_ids"] = ["w1"]
        strong_kinds = ["CONFLICTING_PRIOR_ART"]
    elif relationship == "COMPONENT_ONLY":
        kwargs["component_only_work_ids"] = ["w1"]
        weak_kinds = ["COMPONENT_ONLY"]

    return ScientificClaimEvidenceImpactRecord(
        hypothesis_id="h1",
        claim_id=claim_id,
        novelty_selection_role=role,
        role_precedence={
            "NOVELTY_BEARING": 0,
            "REQUIRED_ENABLING_RELATION": 1,
            "TESTING_PREDICTION": 2,
            "AUXILIARY": 3,
        }[role],
        topology_state="SINGLE_CLAIM_DEGENERATE",
        structural_centrality_score=1.0,
        centrality_rank_within_hypothesis=1,
        highest_centrality_within_hypothesis=True,
        centrality_rank_tied=False,
        classification_coverage_state=(
            "FULLY_CLASSIFIED"
            if classified_fraction == 1.0
            else "PARTIALLY_CLASSIFIED"
        ),
        classification_fraction=classified_fraction,
        evidence_pressure_state=pressure,
        strong_pressure_kinds=strong_kinds,
        strong_pressure_work_ids=strong,
        weak_pressure_kinds=weak_kinds,
        weak_pressure_work_ids=weak,
        strong_pressure_present=bool(strong),
        weak_or_component_signal_present=bool(weak),
        aggregation_order_key=[0.0, 1.0, 0.0, 1.0],
        **kwargs,
    )


def _aggregation(
    graph,
    impact,
    *,
    fully_classified=True,
):
    hyp = ScientificHypothesisEvidenceAggregation(
        hypothesis_id="h1",
        topology_state="SINGLE_CLAIM_DEGENERATE",
        centrality_is_degenerate=True,
        claim_ids=[impact.claim_id],
        novelty_bearing_claim_ids=(
            [impact.claim_id]
            if impact.novelty_selection_role == "NOVELTY_BEARING"
            else []
        ),
        highest_centrality_claim_ids=[impact.claim_id],
        strong_pressure_claim_ids=(
            [impact.claim_id]
            if impact.strong_pressure_present
            else []
        ),
        novelty_bearing_strong_pressure_claim_ids=(
            [impact.claim_id]
            if (
                impact.strong_pressure_present
                and impact.novelty_selection_role == "NOVELTY_BEARING"
            )
            else []
        ),
        highest_centrality_strong_pressure_claim_ids=(
            [impact.claim_id]
            if impact.strong_pressure_present
            else []
        ),
        novelty_bearing_highest_centrality_strong_pressure_claim_ids=(
            [impact.claim_id]
            if (
                impact.strong_pressure_present
                and impact.novelty_selection_role == "NOVELTY_BEARING"
            )
            else []
        ),
        weak_or_component_signal_claim_ids=(
            [impact.claim_id]
            if impact.weak_or_component_signal_present
            else []
        ),
        no_strong_pressure_claim_ids=(
            []
            if impact.strong_pressure_present
            else [impact.claim_id]
        ),
        strong_pressure_work_ids=list(
            impact.strong_pressure_work_ids
        ),
        weak_pressure_work_ids=list(
            impact.weak_pressure_work_ids
        ),
        strong_pressure_kind_counts={
            kind: 1
            for kind in impact.strong_pressure_kinds
        },
        weak_pressure_kind_counts={
            kind: 1
            for kind in impact.weak_pressure_kinds
        },
        evidence_impact_profile=(
            "NOVELTY_BEARING_HIGHEST_CENTRALITY_STRONG_PRESSURE"
            if (
                impact.strong_pressure_present
                and impact.novelty_selection_role == "NOVELTY_BEARING"
            )
            else (
                "HIGHEST_CENTRALITY_STRONG_PRESSURE_OUTSIDE_NOVELTY_BEARING"
                if impact.strong_pressure_present
                else "NO_STRONG_RELATION_PRESSURE"
            )
        ),
        weak_signal_profile=(
            "COMPONENT_CONTEXT_ONLY"
            if impact.weak_or_component_signal_present
            else "NO_WEAK_OR_COMPONENT_SIGNAL"
        ),
        epistemic_coverage_profile=(
            "FULL_CLASSIFICATION_COVERAGE"
            if fully_classified
            else "PARTIAL_CLASSIFICATION_COVERAGE"
        ),
        all_claims_fully_classified=fully_classified,
        any_unclassified_presented_work=not fully_classified,
        review_priority_claim_ids=[impact.claim_id],
    )

    body = {
        "schema_version":
            "scientific-hypothesis-evidence-aggregation-report-v1",
        "source_claim_evidence_graph_id":
            graph.graph_id,
        "source_claim_centrality_report_id":
            "centrality:1",
        "claim_impacts": [
            impact.model_dump(mode="json")
        ],
        "hypothesis_aggregations": [
            hyp.model_dump(mode="json")
        ],
        "claim_count": 1,
        "hypothesis_count": 1,
        "hypothesis_profile_counts": {
            hyp.evidence_impact_profile: 1
        },
        "epistemic_coverage_profile_counts": {
            hyp.epistemic_coverage_profile: 1
        },
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
    digest = _hash_payload(body)
    return ScientificHypothesisEvidenceAggregationReport(
        **body,
        report_id=(
            "scientific_hypothesis_evidence_aggregation:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


def test_directional_counterevidence_on_novelty_bearing_claim_creates_basis_candidate_not_authority():
    claim = _claim(
        relationship_counts={
            "DIRECTIONAL_COUNTEREVIDENCE": 1
        }
    )
    nodes = [
        _claim_node(),
        _work(),
    ]
    edges = [
        _edge(
            relationship="DIRECTIONAL_COUNTEREVIDENCE"
        )
    ]
    graph = _graph(
        claim=claim,
        hypothesis=_hyp(),
        nodes=nodes,
        edges=edges,
    )
    impact = _impact(
        relationship="DIRECTIONAL_COUNTEREVIDENCE"
    )
    aggregation = _aggregation(
        graph,
        impact,
        fully_classified=True,
    )

    report = build_positive_nonobviousness_basis_report(
        graph=graph,
        aggregation=aggregation,
    )

    row = report.claim_records[0]
    summary = report.hypothesis_summaries[0]

    assert row.basis_state == (
        "PRIMARY_POSITIVE_BASIS_CANDIDATE"
    )
    assert row.basis_kinds == [
        "DOCUMENTED_DIRECTIONAL_TENSION"
    ]
    assert (
        summary.future_adjudication_readiness
        == "READY_FOR_POSITIVE_NONOBVIOUSNESS_ADJUDICATION"
    )
    assert (
        report.positive_nonobviousness_authority_created
        is False
    )


def test_component_only_and_no_strong_pressure_never_create_positive_basis():
    claim = _claim(
        relationship_counts={"COMPONENT_ONLY": 1}
    )
    nodes = [
        _claim_node(),
        _work(),
    ]
    edges = [
        _edge(relationship="COMPONENT_ONLY")
    ]
    graph = _graph(
        claim=claim,
        hypothesis=_hyp(),
        nodes=nodes,
        edges=edges,
    )
    impact = _impact(
        relationship="COMPONENT_ONLY"
    )
    aggregation = _aggregation(
        graph,
        impact,
        fully_classified=False,
    )

    report = build_positive_nonobviousness_basis_report(
        graph=graph,
        aggregation=aggregation,
    )

    row = report.claim_records[0]
    summary = report.hypothesis_summaries[0]

    assert row.basis_evidence_count == 0
    assert row.basis_state == (
        "NO_POSITIVE_BASIS_CANDIDATE"
    )
    assert summary.positive_basis_state == (
        "NO_POSITIVE_BASIS_CANDIDATE"
    )
    assert summary.future_adjudication_readiness == (
        "NO_POSITIVE_BASIS"
    )


def test_direct_prior_art_is_blocker_not_positive_basis():
    claim = _claim(
        relationship_counts={"DIRECT_PRIOR_ART": 1}
    )
    nodes = [
        _claim_node(),
        _work(),
    ]
    edges = [
        _edge(relationship="DIRECT_PRIOR_ART")
    ]
    graph = _graph(
        claim=claim,
        hypothesis=_hyp(),
        nodes=nodes,
        edges=edges,
    )
    impact = _impact(
        relationship="DIRECT_PRIOR_ART"
    )
    aggregation = _aggregation(
        graph,
        impact,
        fully_classified=True,
    )

    report = build_positive_nonobviousness_basis_report(
        graph=graph,
        aggregation=aggregation,
    )

    row = report.claim_records[0]
    assert row.basis_evidence_count == 0
    assert row.direct_prior_art_blocker_present is True
    assert report.direct_prior_art_is_not_positive_basis is True


def test_auxiliary_tension_is_recorded_but_does_not_qualify():
    claim = _claim(
        role="AUXILIARY",
        relationship_counts={
            "CONFLICTING_PRIOR_ART": 1
        },
    )
    nodes = [
        _claim_node(role="AUXILIARY"),
        _work(),
    ]
    edges = [
        _edge(
            relationship="CONFLICTING_PRIOR_ART"
        )
    ]
    graph = _graph(
        claim=claim,
        hypothesis=ScientificClaimEvidenceGraphHypothesisSummary(
            hypothesis_id="h1",
            claim_ids=["n1"],
            novelty_bearing_claim_ids=[],
            reviewed_claim_ids=["n1"],
            unreviewed_claim_ids=[],
            claim_count=1,
            reviewed_claim_count=1,
            unreviewed_claim_count=0,
        ),
        nodes=nodes,
        edges=edges,
    )
    impact = _impact(
        role="AUXILIARY",
        relationship="CONFLICTING_PRIOR_ART",
    )
    aggregation = _aggregation(
        graph,
        impact,
        fully_classified=True,
    )

    report = build_positive_nonobviousness_basis_report(
        graph=graph,
        aggregation=aggregation,
    )

    row = report.claim_records[0]
    summary = report.hypothesis_summaries[0]

    assert row.basis_state == (
        "NONQUALIFYING_TENSION_ONLY"
    )
    assert (
        row.qualifying_for_future_nonobviousness_adjudication
        is False
    )
    assert summary.future_adjudication_readiness == (
        "NONQUALIFYING_TENSION_ONLY"
    )


def test_qualifying_basis_with_partial_coverage_stays_epistemically_partial():
    claim = _claim(
        relationship_counts={
            "CONTEXTUAL_CONFLICT": 1
        },
        presented=4,
        classified=1,
    )
    nodes = [
        _claim_node(),
        _work(),
    ]
    edges = [
        _edge(
            relationship="CONTEXTUAL_CONFLICT"
        )
    ]
    graph = _graph(
        claim=claim,
        hypothesis=_hyp(),
        nodes=nodes,
        edges=edges,
    )
    impact = _impact(
        relationship="CONTEXTUAL_CONFLICT",
        classified_fraction=0.25,
    )
    aggregation = _aggregation(
        graph,
        impact,
        fully_classified=False,
    )

    report = build_positive_nonobviousness_basis_report(
        graph=graph,
        aggregation=aggregation,
    )

    summary = report.hypothesis_summaries[0]
    assert summary.positive_basis_state == (
        "QUALIFYING_POSITIVE_BASIS_CANDIDATE_PRESENT"
    )
    assert summary.future_adjudication_readiness == (
        "BASIS_PRESENT_BUT_EPISTEMICALLY_PARTIAL"
    )
    assert (
        report.positive_basis_is_not_nonobviousness_verdict
        is True
    )
