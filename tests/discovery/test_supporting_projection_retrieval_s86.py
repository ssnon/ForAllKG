from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
)
from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
    GroundedFactorProjectionSet,
    GroundedFactorRelationProjection,
    GroundedProjectionFactorBinding,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
    FalsificationCriterion,
)
from pipeline_core.discovery.supporting_projection_retrieval import (
    build_supporting_projection_query_plan,
    build_supporting_projection_retrieval_report,
    build_transport_literature_query_plan,
)


def _portfolio() -> HypothesisPortfolio:
    card = HypothesisCard(
        hypothesis_id="h1",
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="ctx-sha",
        source_report_id="report",
        source_report_sha256="report-sha",
        title="H1",
        hypothesis_statement="A relation.",
        hypothesis_type="cross_evidence_synthesis",
        premise_statement_ids=["p1"],
        inferential_bridge="A bridge.",
        predicted_observations=[
            PredictedObservation(
                observation_id="pred:1",
                observable="signal",
                expected_direction="unspecified",
                rationale="prediction",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="false:1",
                observable="signal",
                falsifying_outcome="no relation",
            )
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )
    return HypothesisPortfolio(
        portfolio_id="portfolio:1",
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="ctx-sha",
        source_report_id="report",
        source_report_sha256="report-sha",
        hypotheses=[card],
    )


def _projection(
    projection_id: str,
    query_variants: list[str],
    *,
    kind: str,
    order: int,
) -> GroundedFactorRelationProjection:
    bindings = []
    if order:
        bindings = [
            GroundedProjectionFactorBinding(
                factor_id="f1",
                group_label="architecture",
                identity_basis_tokens=["architecture"],
                exclusive_identity_basis_tokens=["architecture"],
                exact_source_aliases=["ordered architecture"],
            )
        ][:order]
    return GroundedFactorRelationProjection(
        projection_id=projection_id,
        relation_ir_id="relation:1",
        hypothesis_id="h1",
        claim_id="claim:1",
        projection_kind=kind,
        source_identity_term="architecture",
        retained_factor_ids=[
            binding.factor_id
            for binding in bindings
        ],
        omitted_factor_ids=[],
        factor_order=len(bindings),
        factor_bindings=bindings,
        endpoint_terms=["x", "y"],
        scope_terms=[],
        directional_terms=[],
        canonical_search_terms=["x", "y"],
        canonical_search_query=query_variants[0],
        exact_source_query_variants=query_variants,
        exact_source_query_variant_count=len(query_variants),
        relation_typing_status="READY",
        factorization_status="READY",
        eligible_for_future_typed_retrieval=True,
        reason_codes=[],
    )


def _projection_report() -> GroundedFactorProjectionReport:
    # p1 and p2 intentionally share one literal transport query.
    p1 = _projection(
        "p1",
        ["x y"],
        kind="BASE_RELATION",
        order=0,
    )
    p2 = _projection(
        "p2",
        ["x y"],
        kind="LOWER_ORDER_FACTOR_SUBSET",
        order=0,
    )
    p3 = _projection(
        "p3",
        ["x y ordered architecture"],
        kind="FULL_RELATION",
        order=1,
    )
    pset = GroundedFactorProjectionSet(
        relation_ir_id="relation:1",
        hypothesis_id="h1",
        claim_id="claim:1",
        source_identity_term="architecture",
        factorization_status="READY",
        grounded_factor_count=1,
        max_lower_order_factor_subset_size=2,
        max_projections_per_relation=12,
        max_alias_query_variants_per_projection=4,
        projections=[p1, p2, p3],
        projection_count=3,
        truncated=False,
        planning_ready=True,
        reason_codes=[],
    )
    return GroundedFactorProjectionReport(
        report_id="projection-report:1",
        source_relation_ir_report_id="relation-report:1",
        source_factorization_report_id="factorization-report:1",
        projection_sets=[pset],
        relation_count=1,
        planning_ready_relation_count=1,
        projection_count=3,
        future_retrieval_eligible_projection_count=3,
        truncated_relation_count=0,
    )


def test_supporting_plan_deduplicates_literal_network_queries_only():
    plan = build_supporting_projection_query_plan(
        portfolio=_portfolio(),
        projection_report=_projection_report(),
    )

    assert plan.projection_binding_count == 3
    assert plan.transport_query_count == 2
    assert plan.deduplicated_network_query_count == 1

    shared = next(
        row
        for row in plan.transport_queries
        if row.query_text == "x y"
    )
    assert set(shared.projection_ids) == {"p1", "p2"}
    assert shared.semantic_equivalence_inferred is False


def test_transport_plan_reuses_existing_claim_variant_contract():
    supporting = build_supporting_projection_query_plan(
        portfolio=_portfolio(),
        projection_report=_projection_report(),
    )
    transport = build_transport_literature_query_plan(
        supporting
    )

    assert len(transport.queries) == 2
    assert all(
        row.query_kind == "claim_variant"
        for row in transport.queries
    )
    assert transport.claims == []


def test_retrieval_report_maps_shared_query_results_back_to_each_projection():
    supporting = build_supporting_projection_query_plan(
        portfolio=_portfolio(),
        projection_report=_projection_report(),
    )
    transport = build_transport_literature_query_plan(
        supporting
    )

    shared_query = next(
        row
        for row in supporting.transport_queries
        if row.query_text == "x y"
    )
    other_query = next(
        row
        for row in supporting.transport_queries
        if row.query_text != "x y"
    )

    packet = PriorArtPacket(
        packet_id="packet:1",
        packet_sha256="0" * 64,
        source_portfolio_id="portfolio:1",
        source_query_plan_id=transport.plan_id,
        searched_at_utc="2026-09-22T00:00:00+00:00",
        providers_requested=["crossref"],
        works=[
            PriorArtWork(
                work_id="w1",
                title="Work one",
                doi="10.1/test",
                abstract="abstract one",
                providers=["crossref"],
                retrieval_query_ids=[
                    shared_query.transport_query_id
                ],
                retrieval_claim_ids=["claim:1"],
            ),
            PriorArtWork(
                work_id="w2",
                title="Work two",
                providers=["crossref"],
                retrieval_query_ids=[
                    other_query.transport_query_id
                ],
                retrieval_claim_ids=["claim:1"],
            ),
        ],
        executions=[
            QueryExecution(
                query_id=shared_query.transport_query_id,
                provider="crossref",
                success=True,
                result_count=1,
            ),
            QueryExecution(
                query_id=other_query.transport_query_id,
                provider="crossref",
                success=True,
                result_count=1,
            ),
        ],
        raw_work_count=2,
        canonical_work_count=2,
        deduplicated_work_count=0,
    )

    report = build_supporting_projection_retrieval_report(
        supporting_plan=supporting,
        transport_plan=transport,
        packet=packet,
    )
    by_id = {
        row.projection_id: row
        for row in report.projection_coverage
    }

    assert by_id["p1"].unique_work_count == 1
    assert by_id["p2"].unique_work_count == 1
    assert by_id["p1"].abstract_work_count == 1
    assert by_id["p2"].abstract_work_count == 1
    assert by_id["p3"].unique_work_count == 1
    assert report.relation_adjudication_performed is False
    assert report.absence_based_novelty_authorized is False
