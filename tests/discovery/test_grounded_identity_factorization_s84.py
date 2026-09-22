from __future__ import annotations

from pipeline_core.discovery.grounded_identity_factorization import (
    factorize_relation_identity,
)
from pipeline_core.discovery.reframing.grounded_identity_constituent_shadow import (
    GroundedIdentityAnnotation,
    GroundedIdentityConstituentGroup,
    GroundedIdentityConstituentSpan,
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
        lexical_tokens=text.casefold().replace("-", " ").split(),
        type_labels=["typed"],
        ambiguity_labels=[],
        literal_in_claim_text=True,
        literal_in_required_bridge=(
            None if role == "OBSERVABLE" else True
        ),
    )


def _relation(
    *,
    status: str = "READY",
    identity: str = (
        "architecture-conditioned hotspot accessibility"
    ),
) -> ScientificRelationIR:
    return ScientificRelationIR(
        relation_ir_id="relation:1",
        hypothesis_id="hypothesis:1",
        claim_id="claim:1",
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text="test",
        endpoint_concepts=[
            _concept(
                "e1",
                "RELATION_ENDPOINT",
                "hotspot stability",
                0,
            ),
            _concept(
                "e2",
                "RELATION_ENDPOINT",
                "calibration transfer error",
                1,
            ),
        ],
        identity_concepts=[
            _concept(
                "i1",
                "BRANCH_IDENTITY",
                identity,
                0,
            ),
        ],
        observable_concept=_concept(
            "o1",
            "OBSERVABLE",
            "transfer error",
            0,
        ),
        relation_type_labels=["typed"],
        relation_domain_labels=[],
        relation_scope_features=[],
        typing_status=status,
        reason_codes=[],
    )


def _span(
    *,
    text: str,
    candidate_ref: str,
) -> GroundedIdentityConstituentSpan:
    return GroundedIdentityConstituentSpan(
        candidate_id="candidate:" + candidate_ref,
        candidate_ref=candidate_ref,
        exact_source_text=text,
        matched_source_paths=["scientific_proposal"],
    )


def _annotation() -> GroundedIdentityAnnotation:
    return GroundedIdentityAnnotation(
        claim_id="claim:1",
        identity_term=(
            "architecture-conditioned hotspot accessibility"
        ),
        groups=[
            GroundedIdentityConstituentGroup(
                label="architecture-conditioned",
                spans=[
                    _span(
                        text="architecture-conditioned",
                        candidate_ref="CANDIDATE_01",
                    )
                ],
            ),
            GroundedIdentityConstituentGroup(
                label="hotspot",
                spans=[
                    _span(
                        text="hotspot population",
                        candidate_ref="CANDIDATE_02",
                    )
                ],
            ),
            GroundedIdentityConstituentGroup(
                label="accessibility",
                spans=[
                    _span(
                        text="accessibility",
                        candidate_ref="CANDIDATE_01",
                    )
                ],
            ),
        ],
        rationale="Grounded decomposition.",
    )


def test_grounded_factorization_promotes_three_distinct_source_backed_factors():
    result = factorize_relation_identity(
        relation=_relation(),
        annotation=_annotation(),
    )

    assert result.factorization_status == "READY"
    assert result.grounded_factor_count == 3
    assert result.effective_identity_factor_count == 3
    assert result.complete_identity_token_coverage is True
    assert result.distinct_factor_basis_validated is True

    assert [
        row.identity_basis_tokens
        for row in result.grounded_factors
    ] == [
        ["architecture", "conditioned"],
        ["hotspot"],
        ["accessibility"],
    ]
    assert [
        row.query_aliases
        for row in result.grounded_factors
    ] == [
        ["architecture-conditioned"],
        ["hotspot population"],
        ["accessibility"],
    ]


def test_factor_semantics_exclude_extra_words_from_source_alias():
    annotation = _annotation()
    annotation.groups[0].spans[0].exact_source_text = (
        "ordered substrate architecture-conditioned response"
    )

    result = factorize_relation_identity(
        relation=_relation(),
        annotation=annotation,
    )

    assert result.factorization_status == "READY"
    first = result.grounded_factors[0]
    assert first.identity_basis_tokens == [
        "architecture",
        "conditioned",
    ]
    assert first.query_aliases == [
        "ordered substrate architecture-conditioned response"
    ]


def test_duplicate_factor_basis_fails_closed():
    annotation = GroundedIdentityAnnotation(
        claim_id="claim:1",
        identity_term=(
            "architecture-conditioned hotspot accessibility"
        ),
        groups=[
            GroundedIdentityConstituentGroup(
                label="architecture A",
                spans=[
                    _span(
                        text="architecture-conditioned",
                        candidate_ref="CANDIDATE_01",
                    )
                ],
            ),
            GroundedIdentityConstituentGroup(
                label="architecture B",
                spans=[
                    _span(
                        text="architecture response",
                        candidate_ref="CANDIDATE_02",
                    )
                ],
            ),
            GroundedIdentityConstituentGroup(
                label="hotspot accessibility",
                spans=[
                    _span(
                        text="hotspot accessibility",
                        candidate_ref="CANDIDATE_01",
                    )
                ],
            ),
        ],
        rationale="Bad duplicate grouping.",
    )

    result = factorize_relation_identity(
        relation=_relation(),
        annotation=annotation,
    )

    assert (
        result.factorization_status
        == "INSUFFICIENT_DISTINCT_FACTORS"
    )
    assert result.grounded_factor_count == 0
    assert any(
        code.startswith(
            "grounded_group_has_no_exclusive_identity_basis:"
        )
        for code in result.reason_codes
    )


def test_identity_binding_mismatch_preserves_original_identity():
    annotation = _annotation()
    annotation.identity_term = "different synthetic identity"

    result = factorize_relation_identity(
        relation=_relation(),
        annotation=annotation,
    )

    assert (
        result.factorization_status
        == "IDENTITY_BINDING_MISMATCH"
    )
    assert result.grounded_factor_count == 0
    assert result.effective_identity_factor_count == 1
    assert result.preserved_identity_concept_ids == ["i1"]


def test_nonready_relation_is_never_factorized():
    result = factorize_relation_identity(
        relation=_relation(status="AMBIGUOUS"),
        annotation=_annotation(),
    )

    assert result.factorization_status == "RELATION_NOT_READY"
    assert result.grounded_factor_count == 0
    assert result.preserved_identity_concept_ids == ["i1"]
