from __future__ import annotations

from pipeline_core.discovery.counterevidence_projection_retrieval import (
    build_counterevidence_projection_query_plan,
    build_counterevidence_retrieval_report,
    build_counterevidence_transport_plan,
)
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
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
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


def _factor_binding(index: int) -> GroundedProjectionFactorBinding:
    return GroundedProjectionFactorBinding(
        factor_id=f"f{index}",
        group_label=f"factor_{index}",
        identity_basis_tokens=[f"factor{index}"],
        exclusive_identity_basis_tokens=[f"factor{index}"],
        exact_source_aliases=[f"grounded factor {index}"],
    )


def _projection(
    projection_id: str,
    *,
    kind: str,
    order: int,
    endpoint_terms: list[str] | None = None,
    directional_terms: list[str] | None = None,
) -> GroundedFactorRelationProjection:
    bindings = [
        _factor_binding(index)
        for index in range(1, order + 1)
    ]
    aliases = [
        alias
        for binding in bindings
        for alias in binding.exact_source_aliases
    ]
    endpoint_terms = endpoint_terms or [
        "hotspot stability",
        "transfer error",
    ]

    return GroundedFactorRelationProjection(
        projection_id=projection_id,
        relation_ir_id="relation:1",
        hypothesis_id="h1",
        claim_id="claim:1",
        projection_kind=kind,
        source_identity_term="architecture",
        retained_factor_ids=[
            row.factor_id
            for row in bindings
        ],
        omitted_factor_ids=[],
        factor_order=len(bindings),
        factor_bindings=bindings,
        endpoint_terms=endpoint_terms,
        scope_terms=["across batches"],
        directional_terms=directional_terms or [],
        canonical_search_terms=[
            *endpoint_terms,
            "across batches",
            *aliases,
        ],
        canonical_search_query=" ".join(
            [
                *endpoint_terms,
                "across batches",
                *aliases,
            ]
        ),
        exact_source_query_variants=[
            " ".join(
                [
                    *endpoint_terms,
                    "across batches",
                    *aliases,
                ]
            )
        ],
        exact_source_query_variant_count=1,
        relation_typing_status="READY",
        factorization_status="READY",
        eligible_for_future_typed_retrieval=True,
        reason_codes=[],
    )


def _report(
    projections: list[GroundedFactorRelationProjection],
) -> GroundedFactorProjectionReport:
    pset = GroundedFactorProjectionSet(
        relation_ir_id="relation:1",
        hypothesis_id="h1",
        claim_id="claim:1",
        source_identity_term="architecture",
        factorization_status="READY",
        grounded_factor_count=max(
            (row.factor_order for row in projections),
            default=0,
        ),
        max_lower_order_factor_subset_size=2,
        max_projections_per_relation=12,
        max_alias_query_variants_per_projection=4,
        projections=projections,
        projection_count=len(projections),
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
        projection_count=len(projections),
        future_retrieval_eligible_projection_count=len(projections),
        truncated_relation_count=0,
    )


def test_counterevidence_plan_selects_base_singleton_and_full_by_default():
    projections = [
        _projection("base", kind="BASE_RELATION", order=0),
        _projection(
            "single",
            kind="LOWER_ORDER_FACTOR_SUBSET",
            order=1,
        ),
        _projection(
            "pair",
            kind="LOWER_ORDER_FACTOR_SUBSET",
            order=2,
        ),
        _projection("full", kind="FULL_RELATION", order=2),
    ]

    plan = build_counterevidence_projection_query_plan(
        portfolio=_portfolio(),
        projection_report=_report(projections),
    )

    assert plan.selected_projection_count == 3
    selected_ids = {
        row.projection_id
        for row in plan.bindings
    }
    assert selected_ids == {"base", "single", "full"}
    assert "pair" not in selected_ids


def test_counterevidence_plan_generates_null_independence_and_decoupling():
    plan = build_counterevidence_projection_query_plan(
        portfolio=_portfolio(),
        projection_report=_report(
            [
                _projection("base", kind="BASE_RELATION", order=0),
                _projection("full", kind="FULL_RELATION", order=1),
            ]
        ),
    )

    modes = {
        row.counterevidence_mode
        for row in plan.bindings
    }
    assert {
        "NULL_RELATION",
        "INDEPENDENCE",
        "DECOUPLING",
    } <= modes

    queries = [
        row.query_text
        for row in plan.bindings
    ]
    assert any("no correlation" in row for row in queries)
    assert any("independence" in row for row in queries)
    assert any("decoupling" in row for row in queries)


def test_explicit_direction_generates_opposite_direction_query():
    plan = build_counterevidence_projection_query_plan(
        portfolio=_portfolio(),
        projection_report=_report(
            [
                _projection(
                    "base",
                    kind="BASE_RELATION",
                    order=0,
                    directional_terms=["lower"],
                ),
                _projection("full", kind="FULL_RELATION", order=1),
            ]
        ),
    )

    opposite = [
        row
        for row in plan.bindings
        if row.counterevidence_mode == "OPPOSITE_DIRECTION"
    ]
    assert opposite
    assert all(
        "higher" in row.query_text
        for row in opposite
    )
    assert all(
        "lower" not in row.fixed_scientific_terms
        for row in opposite
    )


def test_heterogeneity_endpoint_adds_distributional_divergence_mode():
    plan = build_counterevidence_projection_query_plan(
        portfolio=_portfolio(),
        projection_report=_report(
            [
                _projection(
                    "base",
                    kind="BASE_RELATION",
                    order=0,
                    endpoint_terms=[
                        "plasmonic environment heterogeneity",
                        "oxidation heterogeneity",
                    ],
                ),
                _projection("full", kind="FULL_RELATION", order=1),
            ]
        ),
    )

    modes = {
        row.counterevidence_mode
        for row in plan.bindings
    }
    assert "DISTRIBUTIONAL_DIVERGENCE" in modes
    distribution_queries = [
        row.query_text
        for row in plan.bindings
        if (
            row.counterevidence_mode
            == "DISTRIBUTIONAL_DIVERGENCE"
        )
    ]
    assert any(
        "homogeneous heterogeneous" in row
        for row in distribution_queries
    )


def test_retrieval_report_does_not_assign_counterevidence_relationship():
    supporting = build_counterevidence_projection_query_plan(
        portfolio=_portfolio(),
        projection_report=_report(
            [
                _projection("base", kind="BASE_RELATION", order=0),
                _projection("full", kind="FULL_RELATION", order=1),
            ]
        ),
    )
    transport = build_counterevidence_transport_plan(
        supporting
    )

    query = transport.queries[0]
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
                title="Potential conflict",
                abstract="A retrieved abstract.",
                retrieval_query_ids=[query.query_id],
                retrieval_claim_ids=["claim:1"],
            )
        ],
        executions=[
            QueryExecution(
                query_id=query.query_id,
                provider="crossref",
                success=True,
                result_count=1,
            )
        ],
        raw_work_count=1,
        canonical_work_count=1,
        deduplicated_work_count=0,
    )

    report = build_counterevidence_retrieval_report(
        counterevidence_plan=supporting,
        transport_plan=transport,
        packet=packet,
    )

    assert report.counterevidence_relationship_assigned is False
    assert report.relation_adjudication_performed is False
    assert report.absence_based_novelty_authorized is False
    assert report.positive_nonobviousness_authority_created is False
