from __future__ import annotations

from pipeline_core.discovery.grounded_factor_projection import (
    compile_grounded_factor_projection_set,
)
from pipeline_core.discovery.grounded_identity_factorization import (
    GroundedIdentityFactor,
    GroundedIdentityFactorSpan,
    RelationIdentityFactorization,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificConceptIR,
    ScientificRelationIR,
)


def _concept(
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
        normalized_text=text.casefold(),
        lexical_tokens=text.casefold().split(),
        type_labels=["typed"],
        ambiguity_labels=[],
        literal_in_claim_text=True,
        literal_in_required_bridge=(
            None if role == "OBSERVABLE" else True
        ),
    )


def _relation(status: str = "READY") -> ScientificRelationIR:
    return ScientificRelationIR(
        relation_ir_id="relation:1",
        hypothesis_id="hypothesis:1",
        claim_id="claim:1",
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text="test",
        endpoint_concepts=[
            _concept("e1", "RELATION_ENDPOINT", "hotspot stability", 0),
            _concept("e2", "RELATION_ENDPOINT", "transfer error", 1),
        ],
        identity_concepts=[
            _concept(
                "i1",
                "BRANCH_IDENTITY",
                "architecture-conditioned hotspot accessibility",
                0,
            )
        ],
        scope_qualifiers=[
            _concept(
                "s1",
                "SCOPE_QUALIFIER",
                "across independent batches",
                0,
            )
        ],
        directional_qualifiers=[
            _concept("d1", "DIRECTIONAL_QUALIFIER", "lower", 0)
        ],
        observable_concept=_concept(
            "o1", "OBSERVABLE", "transfer error", 0
        ),
        relation_type_labels=["typed"],
        relation_domain_labels=[],
        relation_scope_features=[],
        typing_status=status,
        reason_codes=[],
    )


def _factor(
    *,
    factor_id: str,
    label: str,
    basis: list[str],
    aliases: list[str],
) -> GroundedIdentityFactor:
    return GroundedIdentityFactor(
        factor_id=factor_id,
        relation_ir_id="relation:1",
        claim_id="claim:1",
        source_identity_concept_id="i1",
        source_identity_term=(
            "architecture-conditioned hotspot accessibility"
        ),
        group_label=label,
        identity_basis_tokens=basis,
        exclusive_identity_basis_tokens=basis,
        grounded_spans=[
            GroundedIdentityFactorSpan(
                candidate_id="candidate:1",
                candidate_ref="CANDIDATE_01",
                exact_source_text=alias,
                matched_source_paths=["scientific_proposal"],
            )
            for alias in aliases
        ],
        query_aliases=aliases,
    )


def _factorization(
    *,
    status: str = "READY",
) -> RelationIdentityFactorization:
    factors = (
        [
            _factor(
                factor_id="f1",
                label="architecture",
                basis=["architecture", "conditioned"],
                aliases=["architecture-conditioned"],
            ),
            _factor(
                factor_id="f2",
                label="hotspot",
                basis=["hotspot"],
                aliases=["hotspot population", "plasmonic hotspots"],
            ),
            _factor(
                factor_id="f3",
                label="accessibility",
                basis=["accessibility"],
                aliases=["accessibility"],
            ),
        ]
        if status == "READY"
        else []
    )
    return RelationIdentityFactorization(
        relation_ir_id="relation:1",
        hypothesis_id="hypothesis:1",
        claim_id="claim:1",
        factorization_status=status,
        source_identity_concept_id="i1",
        source_identity_term=(
            "architecture-conditioned hotspot accessibility"
        ),
        source_identity_tokens=[
            "architecture",
            "conditioned",
            "hotspot",
            "accessibility",
        ],
        grounded_factors=factors,
        grounded_factor_count=len(factors),
        effective_identity_factor_count=(
            len(factors) if factors else 1
        ),
        preserved_identity_concept_ids=(
            [] if factors else ["i1"]
        ),
        complete_identity_token_coverage=(status == "READY"),
        distinct_factor_basis_validated=(status == "READY"),
        reason_codes=[],
    )


def test_three_grounded_factors_compile_eight_structured_targets():
    result = compile_grounded_factor_projection_set(
        relation=_relation(),
        factorization=_factorization(),
    )

    assert result.planning_ready is True
    assert result.projection_count == 8
    assert [
        row.factor_order
        for row in result.projections
    ] == [0, 1, 1, 1, 2, 2, 2, 3]
    assert result.projections[0].projection_kind == "BASE_RELATION"
    assert result.projections[-1].projection_kind == "FULL_RELATION"


def test_projection_uses_exact_source_alias_not_synthetic_factor_text():
    result = compile_grounded_factor_projection_set(
        relation=_relation(),
        factorization=_factorization(),
    )

    hotspot = next(
        row
        for row in result.projections
        if [
            binding.group_label
            for binding in row.factor_bindings
        ] == ["hotspot"]
    )

    assert "hotspot population" in hotspot.canonical_search_query
    assert (
        "architecture-conditioned hotspot accessibility"
        not in hotspot.canonical_search_query
    )
    assert hotspot.exact_source_query_variant_count == 2
    assert any(
        "plasmonic hotspots" in query
        for query in hotspot.exact_source_query_variants
    )


def test_projection_always_retains_endpoints_scope_and_direction():
    result = compile_grounded_factor_projection_set(
        relation=_relation(),
        factorization=_factorization(),
    )

    for row in result.projections:
        assert row.endpoint_terms == [
            "hotspot stability",
            "transfer error",
        ]
        assert row.scope_terms == [
            "across independent batches"
        ]
        assert row.directional_terms == ["lower"]
        assert "hotspot stability" in row.canonical_search_query
        assert "transfer error" in row.canonical_search_query
        assert "across independent batches" in row.canonical_search_query
        assert "lower" in row.canonical_search_query


def test_not_ready_factorization_fails_closed_without_fake_projections():
    result = compile_grounded_factor_projection_set(
        relation=_relation(),
        factorization=_factorization(
            status="INSUFFICIENT_DISTINCT_FACTORS"
        ),
    )

    assert result.planning_ready is False
    assert result.projection_count == 0
    assert result.projections == []


def test_projection_budget_truncates_but_preserves_full():
    factorization = _factorization()
    factorization.grounded_factors.append(
        _factor(
            factor_id="f4",
            label="fourth",
            basis=["fourth"],
            aliases=["fourth factor"],
        )
    )
    factorization.grounded_factor_count = 4
    factorization.effective_identity_factor_count = 4

    result = compile_grounded_factor_projection_set(
        relation=_relation(),
        factorization=factorization,
        max_lower_order_factor_subset_size=2,
        max_projections_per_relation=5,
    )

    assert result.projection_count == 5
    assert result.truncated is True
    assert result.projections[0].projection_kind == "BASE_RELATION"
    assert result.projections[-1].projection_kind == "FULL_RELATION"
