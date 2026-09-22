from __future__ import annotations

from pipeline_core.discovery.scientific_claim_centrality import (
    build_scientific_claim_centrality_report,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphClaimSummary,
    ScientificClaimEvidenceGraphHypothesisSummary,
    ScientificClaimEvidenceGraphReport,
)


def _graph(
    *,
    claims,
    hypotheses,
):
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
        "nodes": [],
        "edges": [],
        "claim_summaries": [
            row.model_dump(mode="json")
            for row in claims
        ],
        "hypothesis_summaries": [
            row.model_dump(mode="json")
            for row in hypotheses
        ],
        "node_count": 0,
        "edge_count": 0,
        "hypothesis_count": len(hypotheses),
        "claim_count": len(claims),
        "concept_node_count": 0,
        "projection_node_count": 0,
        "work_node_count": 0,
        "presented_work_edge_count": 0,
        "adjudicated_relation_edge_count": 0,
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

    # Reproduce the source graph hash contract.
    import hashlib
    import json

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
    hypothesis_id,
    claim_id,
    concepts,
    role="NOVELTY_BEARING",
    presented=4,
    classified=2,
    relationship_counts=None,
    relation_state="COMPONENTS_OR_CONTEXT_ONLY",
):
    unclassified = presented - classified
    return ScientificClaimEvidenceGraphClaimSummary(
        hypothesis_id=hypothesis_id,
        claim_id=claim_id,
        relation_ir_id="relation:" + claim_id,
        claim_kind="moderator_interaction",
        novelty_selection_role=role,
        typing_status="READY",
        adjudication_status="REVIEWED",
        relation_state=relation_state,
        concept_node_ids=concepts,
        projection_node_ids=["projection:" + claim_id],
        presented_work_node_ids=[
            f"work:{claim_id}:{index}"
            for index in range(presented)
        ],
        classified_work_node_ids=[
            f"work:{claim_id}:{index}"
            for index in range(classified)
        ],
        unclassified_work_node_ids=[
            f"work:{claim_id}:{index}"
            for index in range(classified, presented)
        ],
        presented_work_count=presented,
        classified_work_count=classified,
        unclassified_work_count=unclassified,
        relationship_counts=(
            relationship_counts
            if relationship_counts is not None
            else {"COMPONENT_ONLY": classified}
        ),
    )


def _hypothesis(
    hypothesis_id,
    claim_ids,
):
    return ScientificClaimEvidenceGraphHypothesisSummary(
        hypothesis_id=hypothesis_id,
        claim_ids=claim_ids,
        novelty_bearing_claim_ids=list(claim_ids),
        reviewed_claim_ids=list(claim_ids),
        unreviewed_claim_ids=[],
        claim_count=len(claim_ids),
        reviewed_claim_count=len(claim_ids),
        unreviewed_claim_count=0,
    )


def test_single_claim_hypothesis_is_explicitly_degenerate():
    claim = _claim(
        hypothesis_id="h1",
        claim_id="c1",
        concepts=["a", "b", "c"],
    )
    graph = _graph(
        claims=[claim],
        hypotheses=[_hypothesis("h1", ["c1"])],
    )

    report = build_scientific_claim_centrality_report(
        graph
    )

    row = report.claim_records[0]
    summary = report.hypothesis_summaries[0]

    assert row.topology_state == "SINGLE_CLAIM_DEGENERATE"
    assert row.structural_centrality_score == 1.0
    assert row.centrality_rank_within_hypothesis == 1
    assert summary.centrality_is_degenerate is True
    assert report.single_claim_hypothesis_count == 1


def test_multi_claim_centrality_uses_exact_shared_concept_jaccard():
    claims = [
        _claim(
            hypothesis_id="h1",
            claim_id="c1",
            concepts=["a", "b", "shared"],
        ),
        _claim(
            hypothesis_id="h1",
            claim_id="c2",
            concepts=["c", "shared"],
        ),
        _claim(
            hypothesis_id="h1",
            claim_id="c3",
            concepts=["d", "e"],
        ),
    ]
    graph = _graph(
        claims=claims,
        hypotheses=[
            _hypothesis(
                "h1",
                ["c1", "c2", "c3"],
            )
        ],
    )

    report = build_scientific_claim_centrality_report(
        graph
    )
    by_id = {
        row.claim_id: row
        for row in report.claim_records
    }

    # c1/c2 share one concept; c3 shares none.
    assert by_id["c1"].neighbor_claim_count == 1
    assert by_id["c2"].neighbor_claim_count == 1
    assert by_id["c3"].neighbor_claim_count == 0
    assert by_id["c1"].structural_centrality_score > 0
    assert by_id["c2"].structural_centrality_score > 0
    assert by_id["c3"].structural_centrality_score == 0
    assert (
        report.hypothesis_summaries[0].topology_state
        == "MULTI_CLAIM_CONNECTED"
    )


def test_role_precedence_does_not_change_structural_centrality():
    claims = [
        _claim(
            hypothesis_id="h1",
            claim_id="c1",
            concepts=["a", "shared"],
            role="AUXILIARY",
        ),
        _claim(
            hypothesis_id="h1",
            claim_id="c2",
            concepts=["b", "shared"],
            role="NOVELTY_BEARING",
        ),
    ]
    graph = _graph(
        claims=claims,
        hypotheses=[_hypothesis("h1", ["c1", "c2"])],
    )

    report = build_scientific_claim_centrality_report(
        graph
    )
    by_id = {
        row.claim_id: row
        for row in report.claim_records
    }

    assert (
        by_id["c1"].structural_centrality_score
        == by_id["c2"].structural_centrality_score
    )
    assert by_id["c2"].role_precedence < by_id["c1"].role_precedence
    assert (
        report.role_semantics_kept_separate_from_centrality
        is True
    )


def test_evidence_pressure_uses_classified_subset_but_not_unclassified_as_negative():
    claim = _claim(
        hypothesis_id="h1",
        claim_id="c1",
        concepts=["a", "b"],
        presented=5,
        classified=1,
        relationship_counts={
            "LOWER_ORDER_RELATION_PRIOR_ART": 1
        },
        relation_state="LOWER_ORDER_RELATION_FOUND",
    )
    graph = _graph(
        claims=[claim],
        hypotheses=[_hypothesis("h1", ["c1"])],
    )

    report = build_scientific_claim_centrality_report(
        graph
    )
    row = report.claim_records[0]
    summary = report.hypothesis_summaries[0]

    assert row.evidence_pressure_state == (
        "LOWER_ORDER_PRIOR_ART_PRESSURE"
    )
    assert row.classification_fraction == 0.2
    assert row.classification_coverage_state == (
        "PARTIALLY_CLASSIFIED"
    )
    assert (
        row.evidence_pressure_is_bounded_to_classified_subset
        is True
    )
    assert report.unclassified_is_not_negative_evidence is True
    assert summary.any_unclassified_presented_work is True
    assert summary.aggregation_readiness == (
        "STRUCTURAL_CENTRALITY_READY_EPISTEMICALLY_PARTIAL"
    )


def test_no_strong_signal_does_not_become_novelty_evidence():
    claim = _claim(
        hypothesis_id="h1",
        claim_id="c1",
        concepts=["a"],
        presented=2,
        classified=0,
        relationship_counts={},
        relation_state="NO_MATERIAL_RELATION_SIGNAL",
    )
    graph = _graph(
        claims=[claim],
        hypotheses=[_hypothesis("h1", ["c1"])],
    )

    report = build_scientific_claim_centrality_report(
        graph
    )
    row = report.claim_records[0]

    assert row.evidence_pressure_state == (
        "NO_MATERIAL_SIGNAL_IN_CLASSIFIED_SUBSET"
    )
    assert row.strong_evidence_pressure_present is False
    assert report.absence_based_novelty_authorized is False
    assert (
        report.positive_nonobviousness_authority_created
        is False
    )
    assert (
        report.hypothesis_novelty_aggregation_performed
        is False
    )


def test_multi_claim_without_exact_shared_concepts_is_not_given_fake_centrality():
    claims = [
        _claim(
            hypothesis_id="h1",
            claim_id="c1",
            concepts=["a", "b"],
        ),
        _claim(
            hypothesis_id="h1",
            claim_id="c2",
            concepts=["c", "d"],
        ),
    ]
    graph = _graph(
        claims=claims,
        hypotheses=[_hypothesis("h1", ["c1", "c2"])],
    )

    report = build_scientific_claim_centrality_report(
        graph
    )

    assert (
        report.hypothesis_summaries[0].topology_state
        == "MULTI_CLAIM_NO_EXACT_SHARED_CONCEPTS"
    )
    assert all(
        row.structural_centrality_score == 0.0
        for row in report.claim_records
    )
    assert report.synonym_inference_performed is False
