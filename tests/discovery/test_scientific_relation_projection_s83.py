from __future__ import annotations

from pipeline_core.discovery.scientific_relation_ir import (
    ScientificConceptIR,
    ScientificRelationIR,
)
from pipeline_core.discovery.scientific_relation_projection import (
    compile_relation_projection_set,
)


def _concept(
    *,
    concept_id: str,
    role: str,
    text: str,
    index: int,
) -> ScientificConceptIR:
    return ScientificConceptIR(
        concept_id=concept_id,
        role=role,
        source_field=role.lower(),
        source_index=index,
        surface_text=text,
        normalized_text=text.lower(),
        lexical_tokens=text.lower().split(),
        type_labels=["typed"],
        ambiguity_labels=[],
        literal_in_claim_text=True,
        literal_in_required_bridge=(
            True if role != "OBSERVABLE" else None
        ),
    )


def _relation(
    *,
    identities: list[str],
    status: str = "READY",
) -> ScientificRelationIR:
    endpoints = [
        _concept(
            concept_id="e1",
            role="RELATION_ENDPOINT",
            text="hotspot stability",
            index=0,
        ),
        _concept(
            concept_id="e2",
            role="RELATION_ENDPOINT",
            text="calibration transfer error",
            index=1,
        ),
    ]
    identity_rows = [
        _concept(
            concept_id=f"i{index}",
            role="BRANCH_IDENTITY",
            text=value,
            index=index,
        )
        for index, value in enumerate(identities, start=1)
    ]
    scope = [
        _concept(
            concept_id="s1",
            role="SCOPE_QUALIFIER",
            text="across independent batches",
            index=0,
        )
    ]
    direction = [
        _concept(
            concept_id="d1",
            role="DIRECTIONAL_QUALIFIER",
            text="lower",
            index=0,
        )
    ]
    observable = _concept(
        concept_id="o1",
        role="OBSERVABLE",
        text="transfer error",
        index=0,
    )

    return ScientificRelationIR(
        relation_ir_id="relation:1",
        hypothesis_id="hypothesis:1",
        claim_id="claim:1",
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text="test relation",
        endpoint_concepts=endpoints,
        identity_concepts=identity_rows,
        scope_qualifiers=scope,
        directional_qualifiers=direction,
        observable_concept=observable,
        relation_type_labels=["typed"],
        relation_domain_labels=[],
        relation_scope_features=[],
        typing_status=status,
        reason_codes=[],
    )


def test_three_identity_factors_compile_base_single_pair_and_full():
    result = compile_relation_projection_set(
        _relation(
            identities=[
                "architecture",
                "surface chemistry",
                "analyte access",
            ]
        ),
        max_lower_order_identity_subset_size=2,
        max_projections_per_relation=12,
    )

    # 1 base + 3 singleton + 3 pairwise + 1 full.
    assert result.projection_count == 8
    assert [
        row.identity_order
        for row in result.projections
    ] == [0, 1, 1, 1, 2, 2, 2, 3]

    assert result.projections[0].projection_kind == "BASE_RELATION"
    assert result.projections[-1].projection_kind == "FULL_RELATION"
    assert all(
        row.eligible_for_future_typed_retrieval
        for row in result.projections
    )


def test_projection_always_retains_endpoints_scope_and_direction():
    result = compile_relation_projection_set(
        _relation(
            identities=["architecture", "analyte access"]
        )
    )

    for projection in result.projections:
        assert projection.endpoint_terms == [
            "hotspot stability",
            "calibration transfer error",
        ]
        assert projection.scope_terms == [
            "across independent batches"
        ]
        assert projection.directional_terms == ["lower"]
        assert "hotspot stability" in projection.search_terms
        assert "calibration transfer error" in projection.search_terms
        assert "across independent batches" in projection.search_terms
        assert "lower" in projection.search_terms


def test_single_identity_has_base_and_full_without_fake_middle_projection():
    result = compile_relation_projection_set(
        _relation(
            identities=["architecture-conditioned accessibility"]
        )
    )

    assert result.projection_count == 2
    assert [
        row.projection_kind
        for row in result.projections
    ] == [
        "BASE_RELATION",
        "FULL_RELATION",
    ]
    assert (
        "single_identity_factor_no_proper_nonempty_subset"
        in result.reason_codes
    )


def test_nonready_relation_projections_are_not_future_retrieval_eligible():
    result = compile_relation_projection_set(
        _relation(
            identities=["architecture", "analyte access"],
            status="AMBIGUOUS",
        )
    )

    assert result.projections
    assert all(
        row.eligible_for_future_typed_retrieval is False
        for row in result.projections
    )
    assert (
        "relation_not_ready_for_future_typed_retrieval"
        in result.reason_codes
    )


def test_projection_budget_truncates_deterministically_and_preserves_full():
    result = compile_relation_projection_set(
        _relation(
            identities=[
                "factor A",
                "factor B",
                "factor C",
                "factor D",
            ]
        ),
        max_lower_order_identity_subset_size=2,
        max_projections_per_relation=5,
    )

    assert result.projection_count == 5
    assert result.truncated is True
    assert result.projections[0].projection_kind == "BASE_RELATION"
    assert result.projections[-1].projection_kind == "FULL_RELATION"
    assert (
        "lower_order_projection_budget_truncated"
        in result.reason_codes
    )
