from __future__ import annotations

import pytest

from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
    GroundedFactorProjectionSet,
    GroundedFactorRelationProjection,
)
from pipeline_core.discovery.projection_relation_adjudication import (
    CompiledProjectionRelationMatch,
    ProjectionRelationAdjudicationReport,
    ProjectionRelationCandidateReport,
    ProjectionRelationClaimCandidateSet,
    ProjectionRelationClaimReview,
    ProjectionRelationWorkCandidate,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    build_scientific_claim_evidence_graph,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificConceptIR,
    ScientificRelationIR,
    ScientificRelationIRReport,
)


def _concept(
    cid: str,
    role: str,
    text: str,
    index: int,
):
    return ScientificConceptIR(
        concept_id=cid,
        role=role,
        source_field=role.lower(),
        source_index=index,
        surface_text=text,
        normalized_text=text.casefold(),
        lexical_tokens=text.casefold().split(),
        type_labels=[],
        ambiguity_labels=[],
        literal_in_claim_text=True,
        literal_in_required_bridge=(
            None if role == "OBSERVABLE" else True
        ),
    )


def _relation(
    *,
    claim_id: str,
    hypothesis_id: str = "h1",
    second_endpoint: str = "endpoint B",
):
    return ScientificRelationIR(
        relation_ir_id="relation:" + claim_id,
        hypothesis_id=hypothesis_id,
        claim_id=claim_id,
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text=f"endpoint A relates to {second_endpoint}",
        endpoint_concepts=[
            _concept(
                "concept:" + claim_id + ":a",
                "RELATION_ENDPOINT",
                "endpoint A",
                0,
            ),
            _concept(
                "concept:" + claim_id + ":b",
                "RELATION_ENDPOINT",
                second_endpoint,
                1,
            ),
        ],
        identity_concepts=[
            _concept(
                "concept:" + claim_id + ":identity",
                "BRANCH_IDENTITY",
                "identity C",
                0,
            ),
        ],
        observable_concept=_concept(
            "concept:" + claim_id + ":observable",
            "OBSERVABLE",
            second_endpoint,
            0,
        ),
        relation_type_labels=[],
        relation_domain_labels=[],
        relation_scope_features=[],
        typing_status="READY",
        reason_codes=[],
    )


def _projection(
    claim_id: str,
    hypothesis_id: str = "h1",
):
    return GroundedFactorRelationProjection(
        projection_id="projection:" + claim_id,
        relation_ir_id="relation:" + claim_id,
        hypothesis_id=hypothesis_id,
        claim_id=claim_id,
        projection_kind="BASE_RELATION",
        source_identity_term="identity C",
        retained_factor_ids=[],
        omitted_factor_ids=["factor:c"],
        factor_order=0,
        factor_bindings=[],
        endpoint_terms=["endpoint A", "endpoint B"],
        scope_terms=[],
        directional_terms=[],
        canonical_search_terms=["endpoint A", "endpoint B"],
        canonical_search_query="endpoint A endpoint B",
        exact_source_query_variants=["endpoint A endpoint B"],
        exact_source_query_variant_count=1,
        relation_typing_status="READY",
        factorization_status="READY",
        eligible_for_future_typed_retrieval=True,
        reason_codes=[],
    )


def _candidate(
    claim_id: str,
    work_id: str,
):
    return ProjectionRelationWorkCandidate(
        review_work_id=work_id,
        source_work_ids=["source:" + work_id],
        title="Example work " + work_id,
        year=2020,
        abstract="Endpoint A is directly associated with endpoint B.",
        typed_compatibility_state="DOMAIN_COMPATIBLE",
        endpoint_lexical_coverage=1.0,
        endpoint_discriminative_coverages=[1.0, 1.0],
        endpoint_discriminative_match_counts=[2, 2],
        endpoint_required_match_counts=[2, 2],
        endpoint_supported_count=2,
        all_endpoints_supported=True,
        relation_anchor_tier="PAIR_OR_MULTI_ENDPOINT_ANCHORED",
        supporting_projection_ids=["projection:" + claim_id],
        supporting_projection_kinds=["BASE_RELATION"],
        supporting_factor_orders=[0],
        selected_lanes=["SUPPORTING_PRIOR_ART"],
        selection_score=20.0,
    )


def _inputs():
    relations = [
        _relation(claim_id="claim:1"),
        _relation(
            claim_id="claim:2",
            second_endpoint="endpoint D",
        ),
    ]
    relation_report = ScientificRelationIRReport(
        report_id="relation-report:1",
        source_atomic_report_id="atomic-report:1",
        domain_profile_id="test",
        typing_adapter_id="test",
        relations=relations,
        relation_count=2,
        ready_count=2,
        partial_count=0,
        ambiguous_count=0,
        structurally_invalid_count=0,
    )

    projections = [
        _projection("claim:1"),
        _projection("claim:2"),
    ]
    projection_report = GroundedFactorProjectionReport(
        report_id="projection-report:1",
        source_relation_ir_report_id="relation-report:1",
        source_factorization_report_id="factor-report:1",
        projection_sets=[
            GroundedFactorProjectionSet(
                relation_ir_id="relation:claim:1",
                hypothesis_id="h1",
                claim_id="claim:1",
                source_identity_term="identity C",
                factorization_status="READY",
                grounded_factor_count=1,
                max_lower_order_factor_subset_size=2,
                max_projections_per_relation=12,
                max_alias_query_variants_per_projection=4,
                projections=[projections[0]],
                projection_count=1,
                truncated=False,
                planning_ready=False,
                reason_codes=[],
            ),
            GroundedFactorProjectionSet(
                relation_ir_id="relation:claim:2",
                hypothesis_id="h1",
                claim_id="claim:2",
                source_identity_term="identity C",
                factorization_status="READY",
                grounded_factor_count=1,
                max_lower_order_factor_subset_size=2,
                max_projections_per_relation=12,
                max_alias_query_variants_per_projection=4,
                projections=[projections[1]],
                projection_count=1,
                truncated=False,
                planning_ready=False,
                reason_codes=[],
            ),
        ],
        relation_count=2,
        planning_ready_relation_count=0,
        projection_count=2,
        future_retrieval_eligible_projection_count=2,
        truncated_relation_count=0,
    )

    c11 = _candidate("claim:1", "work:shared")
    c12 = _candidate("claim:1", "work:unclassified")
    c21 = _candidate("claim:2", "work:shared")

    candidate_report = ProjectionRelationCandidateReport(
        report_id="candidate-report:1",
        source_relation_ir_report_id="relation-report:1",
        source_projection_report_id="projection-report:1",
        source_supporting_query_plan_id="support-plan:1",
        source_supporting_prior_art_packet_id="support-packet:1",
        source_counterevidence_query_plan_id="counter-plan:1",
        source_counterevidence_prior_art_packet_id="counter-packet:1",
        claims=[
            ProjectionRelationClaimCandidateSet(
                hypothesis_id="h1",
                claim_id="claim:1",
                relation_ir_id="relation:claim:1",
                claim_text=relations[0].claim_text,
                endpoint_terms=["endpoint A", "endpoint B"],
                identity_terms=["identity C"],
                projection_ids=["projection:claim:1"],
                full_projection_ids=[],
                lower_order_projection_ids=["projection:claim:1"],
                candidates=[c11, c12],
                candidate_count=2,
                abstract_candidate_count=2,
                typed_identity_excluded_work_count=0,
                source_unique_work_count=2,
                max_review_works=20,
            ),
            ProjectionRelationClaimCandidateSet(
                hypothesis_id="h1",
                claim_id="claim:2",
                relation_ir_id="relation:claim:2",
                claim_text=relations[1].claim_text,
                endpoint_terms=["endpoint A", "endpoint D"],
                identity_terms=["identity C"],
                projection_ids=["projection:claim:2"],
                full_projection_ids=[],
                lower_order_projection_ids=["projection:claim:2"],
                candidates=[c21],
                candidate_count=1,
                abstract_candidate_count=1,
                typed_identity_excluded_work_count=0,
                source_unique_work_count=1,
                max_review_works=20,
            ),
        ],
        claim_count=2,
        total_review_candidate_count=3,
    )

    match1 = CompiledProjectionRelationMatch(
        work_id="work:shared",
        source_work_ids=["source:work:shared"],
        relationship="LOWER_ORDER_RELATION_PRIOR_ART",
        original_relationship="LOWER_ORDER_RELATION_PRIOR_ART",
        confidence=0.9,
        basis_projection_ids=["projection:claim:1"],
        evidence_span=(
            "Endpoint A is directly associated with endpoint B"
        ),
        rationale="lower-order relation",
        typed_compatibility_state="DOMAIN_COMPATIBLE",
        abstract_available=True,
    )
    match2 = CompiledProjectionRelationMatch(
        work_id="work:shared",
        source_work_ids=["source:work:shared"],
        relationship="COMPONENT_ONLY",
        original_relationship="COMPONENT_ONLY",
        confidence=0.8,
        basis_projection_ids=[],
        evidence_span="",
        rationale="component context only",
        typed_compatibility_state="DOMAIN_COMPATIBLE",
        abstract_available=True,
    )

    reviews = [
        ProjectionRelationClaimReview(
            hypothesis_id="h1",
            claim_id="claim:1",
            relation_ir_id="relation:claim:1",
            relation_state="LOWER_ORDER_RELATION_FOUND",
            matches=[match1],
            presented_work_count=2,
            classified_work_count=1,
            unclassified_work_ids=["work:unclassified"],
            reviewed_work_count=1,
            direct_prior_art_work_ids=[],
            partial_prior_art_work_ids=[],
            lower_order_prior_art_work_ids=["work:shared"],
            directional_counterevidence_work_ids=[],
            contextual_conflict_work_ids=[],
            conflicting_prior_art_work_ids=[],
            component_only_work_ids=[],
            reviewer_unknown_work_ids=[],
            interpretation="lower order found",
        ),
        ProjectionRelationClaimReview(
            hypothesis_id="h1",
            claim_id="claim:2",
            relation_ir_id="relation:claim:2",
            relation_state="COMPONENTS_OR_CONTEXT_ONLY",
            matches=[match2],
            presented_work_count=1,
            classified_work_count=1,
            unclassified_work_ids=[],
            reviewed_work_count=1,
            direct_prior_art_work_ids=[],
            partial_prior_art_work_ids=[],
            lower_order_prior_art_work_ids=[],
            directional_counterevidence_work_ids=[],
            contextual_conflict_work_ids=[],
            conflicting_prior_art_work_ids=[],
            component_only_work_ids=["work:shared"],
            reviewer_unknown_work_ids=[],
            interpretation="component only",
        ),
    ]

    adjudication_report = ProjectionRelationAdjudicationReport(
        report_id="adjudication-report:1",
        source_candidate_report_id="candidate-report:1",
        backend_name="test",
        model_name="test",
        reviews=reviews,
        reviewed_claim_count=2,
        presented_work_count=3,
        classified_work_count=2,
        unclassified_work_count=1,
        reviewed_work_count=2,
        llm_calls_performed=2,
        direct_signal_work_count=0,
        lower_order_signal_work_count=1,
        counterevidence_signal_work_count=0,
        conflicting_signal_work_count=0,
    )

    return (
        relation_report,
        projection_report,
        candidate_report,
        adjudication_report,
    )


def test_claim_evidence_graph_deduplicates_shared_work_and_exact_concepts():
    (
        relation_report,
        projection_report,
        candidate_report,
        adjudication_report,
    ) = _inputs()

    graph = build_scientific_claim_evidence_graph(
        relation_ir_report=relation_report,
        projection_report=projection_report,
        candidate_report=candidate_report,
        adjudication_report=adjudication_report,
    )

    work_nodes = [
        row
        for row in graph.nodes
        if row.node_kind == "WORK"
    ]
    assert len(work_nodes) == 2

    endpoint_a_nodes = [
        row
        for row in graph.nodes
        if (
            row.node_kind == "CONCEPT"
            and row.normalized_text == "endpoint a"
        )
    ]
    assert len(endpoint_a_nodes) == 1
    assert endpoint_a_nodes[0].claim_ids == [
        "claim:1",
        "claim:2",
    ]

    assert graph.hypothesis_count == 1
    assert graph.claim_count == 2


def test_unclassified_work_is_presentation_only_not_negative_evidence():
    args = _inputs()
    graph = build_scientific_claim_evidence_graph(
        relation_ir_report=args[0],
        projection_report=args[1],
        candidate_report=args[2],
        adjudication_report=args[3],
    )

    unclassified_edges = [
        row
        for row in graph.edges
        if (
            row.edge_kind == "CLAIM_PRESENTED_WORK"
            and row.claim_id == "claim:1"
            and row.review_status == "UNCLASSIFIED"
        )
    ]
    assert len(unclassified_edges) == 1

    work_node_id = unclassified_edges[0].target_node_id
    assert not any(
        row.edge_kind == "WORK_ADJUDICATED_TO_CLAIM"
        and row.claim_id == "claim:1"
        and row.source_node_id == work_node_id
        for row in graph.edges
    )
    assert graph.unclassified_is_not_negative_evidence is True
    assert graph.absence_based_novelty_authorized is False


def test_strong_adjudication_edge_preserves_exact_span_and_projection_basis():
    args = _inputs()
    graph = build_scientific_claim_evidence_graph(
        relation_ir_report=args[0],
        projection_report=args[1],
        candidate_report=args[2],
        adjudication_report=args[3],
    )

    edges = [
        row
        for row in graph.edges
        if (
            row.edge_kind == "WORK_ADJUDICATED_TO_CLAIM"
            and row.claim_id == "claim:1"
        )
    ]
    assert len(edges) == 1
    assert edges[0].relationship == (
        "LOWER_ORDER_RELATION_PRIOR_ART"
    )
    assert (
        "Endpoint A is directly associated with endpoint B"
        in edges[0].evidence_span
    )

    basis = [
        row
        for row in graph.edges
        if (
            row.edge_kind == "WORK_BASIS_PROJECTION"
            and row.claim_id == "claim:1"
        )
    ]
    assert len(basis) == 1
    assert basis[0].basis_projection_id == (
        "projection:claim:1"
    )


def test_graph_fails_closed_on_source_provenance_mismatch():
    args = list(_inputs())
    bad_candidate_report = args[2].model_copy(
        update={
            "source_projection_report_id":
                "projection-report:wrong"
        }
    )

    with pytest.raises(
        ValueError,
        match="candidate report does not derive",
    ):
        build_scientific_claim_evidence_graph(
            relation_ir_report=args[0],
            projection_report=args[1],
            candidate_report=bad_candidate_report,
            adjudication_report=args[3],
        )


def test_graph_does_not_create_centrality_or_novelty_authority():
    args = _inputs()
    graph = build_scientific_claim_evidence_graph(
        relation_ir_report=args[0],
        projection_report=args[1],
        candidate_report=args[2],
        adjudication_report=args[3],
    )

    assert graph.centrality_scoring_performed is False
    assert graph.aggregation_performed is False
    assert (
        graph.positive_nonobviousness_authority_created
        is False
    )
    assert graph.production_selection_changed is False
