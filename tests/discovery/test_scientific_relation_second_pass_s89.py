from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
    GroundedFactorProjectionSet,
    GroundedFactorRelationProjection,
    GroundedProjectionFactorBinding,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificConceptIR,
    ScientificRelationIR,
    ScientificRelationIRReport,
)
from pipeline_core.discovery.scientific_relation_second_pass import (
    EndpointSemanticExpansion,
    RelationSemanticExpansionDraft,
    _base_doi,
    build_resolution_targets,
    build_second_pass_plan,
)


class _Novelty:
    domain_patterns = (("SERS", (r"sers",)),)


class _Profile:
    profile_id = "test"
    description = "test SERS profile"
    novelty = _Novelty()


def _concept(cid: str, role: str, text: str, index: int):
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


def _relation() -> ScientificRelationIR:
    return ScientificRelationIR(
        relation_ir_id="relation:1",
        hypothesis_id="h1",
        claim_id="claim:1",
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text="endpoint one relates to endpoint two",
        endpoint_concepts=[
            _concept("e1", "RELATION_ENDPOINT", "plasmonic environment heterogeneity", 0),
            _concept("e2", "RELATION_ENDPOINT", "reporter oxidation heterogeneity", 1),
        ],
        identity_concepts=[
            _concept("i1", "BRANCH_IDENTITY", "hotspot accessibility", 0),
        ],
        observable_concept=_concept("o1", "OBSERVABLE", "oxidation", 0),
        relation_type_labels=[],
        relation_domain_labels=[],
        relation_scope_features=[],
        typing_status="READY",
        reason_codes=[],
    )


def _projection(pid: str, kind: str, order: int):
    bindings = []
    if order:
        bindings = [
            GroundedProjectionFactorBinding(
                factor_id="f1",
                group_label="hotspot",
                identity_basis_tokens=["hotspot"],
                exclusive_identity_basis_tokens=["hotspot"],
                exact_source_aliases=["hotspot population"],
            )
        ]
    return GroundedFactorRelationProjection(
        projection_id=pid,
        relation_ir_id="relation:1",
        hypothesis_id="h1",
        claim_id="claim:1",
        projection_kind=kind,
        source_identity_term="hotspot accessibility",
        retained_factor_ids=[b.factor_id for b in bindings],
        omitted_factor_ids=[],
        factor_order=len(bindings),
        factor_bindings=bindings,
        endpoint_terms=[
            "plasmonic environment heterogeneity",
            "reporter oxidation heterogeneity",
        ],
        scope_terms=[],
        directional_terms=[],
        canonical_search_terms=["a", "b"],
        canonical_search_query="a b",
        exact_source_query_variants=["a b"],
        exact_source_query_variant_count=1,
        relation_typing_status="READY",
        factorization_status="READY",
        eligible_for_future_typed_retrieval=True,
        reason_codes=[],
    )


def _projection_report():
    rows = [
        _projection("p-base", "BASE_RELATION", 0),
        _projection("p-full", "FULL_RELATION", 1),
    ]
    pset = GroundedFactorProjectionSet(
        relation_ir_id="relation:1",
        hypothesis_id="h1",
        claim_id="claim:1",
        source_identity_term="hotspot accessibility",
        factorization_status="READY",
        grounded_factor_count=1,
        max_lower_order_factor_subset_size=2,
        max_projections_per_relation=12,
        max_alias_query_variants_per_projection=4,
        projections=rows,
        projection_count=2,
        truncated=False,
        planning_ready=True,
        reason_codes=[],
    )
    return GroundedFactorProjectionReport(
        report_id="projection-report:1",
        source_relation_ir_report_id="relation-report:1",
        source_factorization_report_id="factor-report:1",
        projection_sets=[pset],
        relation_count=1,
        planning_ready_relation_count=1,
        projection_count=2,
        future_retrieval_eligible_projection_count=2,
        truncated_relation_count=0,
    )


def test_second_pass_compiles_semantic_queries_without_authority():
    relation = _relation()
    relation_report = ScientificRelationIRReport(
        report_id="relation-report:1",
        source_atomic_report_id="atomic-report:1",
        domain_profile_id="test",
        typing_adapter_id="test",
        relations=[relation],
        relation_count=1,
        ready_count=1,
        partial_count=0,
        ambiguous_count=0,
        structurally_invalid_count=0,
    )
    draft = RelationSemanticExpansionDraft(
        endpoint_expansions=[
            EndpointSemanticExpansion(
                endpoint_concept_id="e1",
                source_text="plasmonic environment heterogeneity",
                retrieval_aliases=["optical enhancement heterogeneity"],
            ),
            EndpointSemanticExpansion(
                endpoint_concept_id="e2",
                source_text="reporter oxidation heterogeneity",
                retrieval_aliases=["surface reaction heterogeneity"],
            ),
        ],
        counterevidence_phrases=[
            "uniform optical response heterogeneous reaction activity"
        ],
    )

    plan = build_second_pass_plan(
        relation_ir_report=relation_report,
        projection_report=_projection_report(),
        drafts_by_claim_id={"claim:1": draft},
        domain_profile=_Profile(),
        source_portfolio_id="portfolio:1",
        max_queries_per_claim=6,
    )

    assert plan.query_count >= 2
    assert plan.query_count <= 6
    assert all(row.novelty_authority is False for row in plan.bindings)
    assert any("SERS" in row.query_text for row in plan.bindings)
    assert any(
        row.role == "SEMANTIC_COUNTEREVIDENCE"
        for row in plan.bindings
    )


def test_invalid_endpoint_binding_does_not_promote_generated_alias():
    relation = _relation()
    relation_report = ScientificRelationIRReport(
        report_id="relation-report:1",
        source_atomic_report_id="atomic-report:1",
        domain_profile_id="test",
        typing_adapter_id="test",
        relations=[relation],
        relation_count=1,
        ready_count=1,
        partial_count=0,
        ambiguous_count=0,
        structurally_invalid_count=0,
    )
    draft = RelationSemanticExpansionDraft(
        endpoint_expansions=[
            EndpointSemanticExpansion(
                endpoint_concept_id="e1",
                source_text="WRONG SOURCE",
                retrieval_aliases=["invented alias"],
            )
        ]
    )

    plan = build_second_pass_plan(
        relation_ir_report=relation_report,
        projection_report=_projection_report(),
        drafts_by_claim_id={"claim:1": draft},
        domain_profile=_Profile(),
        source_portfolio_id="portfolio:1",
        max_queries_per_claim=4,
    )

    assert all(
        "invented alias" not in row.query_text
        for row in plan.bindings
    )


def test_supplementary_doi_is_reduced_to_base_family_for_resolution():
    assert _base_doi("10.1234/example.s001") == "10.1234/example"


def test_resolution_targets_retry_missing_abstract_by_base_doi():
    packet = PriorArtPacket(
        packet_id="packet:1",
        packet_sha256="0" * 64,
        source_portfolio_id="portfolio:1",
        source_query_plan_id="plan:1",
        searched_at_utc="2026-09-22T00:00:00+00:00",
        providers_requested=["crossref"],
        works=[
            PriorArtWork(
                work_id="w1",
                title="Example supplementary record",
                doi="10.1234/example.s001",
                abstract=None,
                retrieval_query_ids=["q1"],
                retrieval_claim_ids=["claim:1"],
            )
        ],
        executions=[],
        raw_work_count=1,
        canonical_work_count=1,
        deduplicated_work_count=0,
    )

    targets = build_resolution_targets(packet, max_queries=10)
    assert any(
        row.reason == "SUPPLEMENTARY_DOI_BASE_LOOKUP"
        and row.query_text == "10.1234/example"
        for row in targets
    )



def test_resolution_identity_filter_drops_unrelated_provider_hits():
    from pipeline_core.discovery.external_novelty_contracts import (
        LiteratureQuery,
        LiteratureQueryPlan,
    )
    from pipeline_core.discovery.scientific_relation_second_pass import (
        SecondPassResolutionTarget,
        filter_resolution_packet_to_source_identities,
    )

    semantic = PriorArtPacket(
        packet_id="semantic:1",
        packet_sha256="0" * 64,
        source_portfolio_id="portfolio:1",
        source_query_plan_id="semantic-plan:1",
        searched_at_utc="2026-09-22T00:00:00+00:00",
        providers_requested=["crossref"],
        works=[
            PriorArtWork(
                work_id="source",
                title="Target work",
                doi="10.1234/target.s001",
                abstract=None,
                retrieval_query_ids=["semantic-q"],
                retrieval_claim_ids=["claim:1"],
            )
        ],
        executions=[],
        raw_work_count=1,
        canonical_work_count=1,
        deduplicated_work_count=0,
    )

    query = LiteratureQuery(
        query_id="resolution-q",
        hypothesis_id="resolution:source",
        claim_id="claim:1",
        query_kind="claim_exact_verification",
        query_text="10.1234/target",
    )
    transport = LiteratureQueryPlan(
        plan_id="resolution-plan",
        plan_sha256="0" * 64,
        source_portfolio_id="portfolio:1",
        queries=[query],
        claims=[],
    )
    target = SecondPassResolutionTarget(
        source_work_id="source",
        claim_id="claim:1",
        reason="SUPPLEMENTARY_DOI_BASE_LOOKUP",
        query_text="10.1234/target",
    )

    resolution = PriorArtPacket(
        packet_id="resolution:1",
        packet_sha256="0" * 64,
        source_portfolio_id="portfolio:1",
        source_query_plan_id="resolution-plan",
        searched_at_utc="2026-09-22T00:00:00+00:00",
        providers_requested=["crossref"],
        works=[
            PriorArtWork(
                work_id="good",
                title="Target work",
                doi="10.1234/target",
                abstract="resolved abstract",
                retrieval_query_ids=["resolution-q"],
                retrieval_claim_ids=["claim:1"],
            ),
            PriorArtWork(
                work_id="bad",
                title="Unrelated result returned by broad DOI search",
                doi="10.9999/unrelated",
                abstract="wrong abstract",
                retrieval_query_ids=["resolution-q"],
                retrieval_claim_ids=["claim:1"],
            ),
        ],
        executions=[],
        raw_work_count=2,
        canonical_work_count=2,
        deduplicated_work_count=0,
    )

    filtered = filter_resolution_packet_to_source_identities(
        semantic_packet=semantic,
        resolution_packet=resolution,
        resolution_plan=transport,
        resolution_targets=[target],
    )

    assert [work.work_id for work in filtered.works] == ["good"]



def test_semantic_query_budget_reserves_full_and_counter_roles_under_alias_saturation():
    relation = _relation()
    relation_report = ScientificRelationIRReport(
        report_id="relation-report:role-budget",
        source_atomic_report_id="atomic-report:role-budget",
        domain_profile_id="test",
        typing_adapter_id="test",
        relations=[relation],
        relation_count=1,
        ready_count=1,
        partial_count=0,
        ambiguous_count=0,
        structurally_invalid_count=0,
    )
    draft = RelationSemanticExpansionDraft(
        endpoint_expansions=[
            EndpointSemanticExpansion(
                endpoint_concept_id="e1",
                source_text="plasmonic environment heterogeneity",
                retrieval_aliases=[
                    "optical enhancement heterogeneity",
                    "plasmonic field variation",
                    "local field nonuniformity",
                ],
            ),
            EndpointSemanticExpansion(
                endpoint_concept_id="e2",
                source_text="reporter oxidation heterogeneity",
                retrieval_aliases=[
                    "surface reaction heterogeneity",
                    "reporter redox variation",
                    "chemical activity nonuniformity",
                ],
            ),
        ],
        counterevidence_phrases=[
            "decoupled optical and chemical response",
            "uniform optical response heterogeneous chemistry",
            "no correlation between optical and chemical activity",
        ],
    )

    plan = build_second_pass_plan(
        relation_ir_report=relation_report,
        projection_report=_projection_report(),
        drafts_by_claim_id={"claim:1": draft},
        domain_profile=_Profile(),
        source_portfolio_id="portfolio:1",
        max_queries_per_claim=8,
    )

    roles = [row.role for row in plan.bindings]

    assert len(roles) == 8
    assert roles.count("SEMANTIC_ENDPOINT_PAIR") == 5
    assert roles.count("SEMANTIC_FULL_RELATION") == 1
    assert roles.count("SEMANTIC_COUNTEREVIDENCE") == 2
