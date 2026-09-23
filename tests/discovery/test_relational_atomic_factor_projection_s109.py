from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)
from pipeline_core.discovery.relational_atomic_factor_projection import (
    compile_relational_atomic_factor_projection_set,
    factorize_relational_atomic_identity,
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
    identities: list[str] | None = None,
) -> ScientificRelationIR:
    identities = identities or ["substrate composition"]
    return ScientificRelationIR(
        relation_ir_id="relation:1",
        hypothesis_id="hypothesis:final",
        claim_id="claim:1",
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text=(
            "Substrate composition changes SERS enhancement factor "
            "sensitivity to nanostructure spacing."
        ),
        endpoint_concepts=[
            _concept(
                "e1",
                "RELATION_ENDPOINT",
                "SERS enhancement factor",
                0,
            ),
            _concept(
                "e2",
                "RELATION_ENDPOINT",
                "nanostructure spacing",
                1,
            ),
        ],
        identity_concepts=[
            _concept(
                f"i{index}",
                "BRANCH_IDENTITY",
                value,
                index,
            )
            for index, value in enumerate(identities, start=1)
        ],
        observable_concept=_concept(
            "o1",
            "OBSERVABLE",
            "SERS enhancement factor",
            0,
        ),
        relation_type_labels=[
            "sers_enhancement_metric",
            "nanostructure_design_variable",
            "material_composition_variable",
        ],
        relation_domain_labels=["SERS"],
        relation_scope_features=[],
        typing_status=status,
        reason_codes=[],
        source_contract="relational-atomic-projection-report-v1",
    )


def _candidate(
    *,
    statement: str = (
        "Substrate composition moderates the response of SERS enhancement "
        "factor to nanostructure spacing."
    ),
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="hypothesis:candidate",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="c" * 64,
        source_report_id="report:1",
        source_report_sha256="r" * 64,
        title="Composition-conditioned SERS response",
        hypothesis_statement=statement,
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        inferential_bridge=statement,
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable=(
                    "The SERS enhancement-factor response differs "
                    "between substrate compositions."
                ),
                expected_direction="unspecified",
                rationale="Composition is the moderator.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable=(
                    "The SERS enhancement-factor response differs "
                    "between substrate compositions."
                ),
                falsifying_outcome=(
                    "The response is indistinguishable between "
                    "substrate compositions."
                ),
            )
        ],
        assumptions=[],
        source_paper_ids=["paper:1"],
        gap_paper_ids=[],
        cross_paper_synthesis=False,
        candidate_dependency="none",
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def test_atomic_identity_is_preserved_whole_from_exact_candidate_source() -> None:
    result = factorize_relational_atomic_identity(
        relation=_relation(),
        candidate=_candidate(),
        candidate_hypothesis_id="hypothesis:candidate",
    )

    assert result.factorization_status == "ATOMIC_IDENTITY_READY"
    assert result.source_identity_term == "substrate composition"
    assert result.identity_basis_tokens == ["substrate", "composition"]
    assert result.exact_source_aliases == ["Substrate composition"]
    assert result.factor_id is not None
    assert result.synthetic_constituent_decomposition_performed is False
    assert any(
        "hypothesis_statement" in path
        for span in result.source_spans
        for path in span.matched_source_paths
    )


def test_atomic_identity_missing_from_candidate_source_fails_closed() -> None:
    result = factorize_relational_atomic_identity(
        relation=_relation(),
        candidate=_candidate(
            statement=(
                "Material choice moderates the SERS response to "
                "nanostructure spacing."
            )
        ),
        candidate_hypothesis_id="hypothesis:candidate",
    )

    # The prediction/falsifier use plural "substrate compositions"; exact
    # singular atomic identity grounding must not silently stem or paraphrase.
    assert result.factorization_status == "SOURCE_GROUNDING_MISSING"
    assert result.factor_id is None
    assert result.exact_source_aliases == []
    assert result.reason_codes == [
        "atomic_identity_not_exact_in_candidate_hypothesis_source"
    ]


def test_multiple_identity_concepts_are_not_silently_collapsed() -> None:
    result = factorize_relational_atomic_identity(
        relation=_relation(
            identities=["substrate composition", "analyte identity"]
        ),
        candidate=_candidate(),
        candidate_hypothesis_id="hypothesis:candidate",
    )

    assert result.factorization_status == (
        "IDENTITY_CARDINALITY_UNSUPPORTED"
    )
    assert result.factor_id is None


def test_single_atomic_identity_compiles_only_base_and_full() -> None:
    relation = _relation()
    factorization = factorize_relational_atomic_identity(
        relation=relation,
        candidate=_candidate(),
        candidate_hypothesis_id="hypothesis:candidate",
    )
    projections = compile_relational_atomic_factor_projection_set(
        relation=relation,
        factorization=factorization,
    )

    assert projections.planning_ready is True
    assert projections.grounded_factor_count == 1
    assert projections.projection_count == 2
    assert [
        row.projection_kind for row in projections.projections
    ] == ["BASE_RELATION", "FULL_RELATION"]
    assert [
        row.factor_order for row in projections.projections
    ] == [0, 1]

    base, full = projections.projections
    assert "Substrate composition" not in base.canonical_search_query
    assert "Substrate composition" in full.canonical_search_query
    assert base.eligible_for_future_typed_retrieval is True
    assert full.eligible_for_future_typed_retrieval is True


def test_nonready_relation_never_reaches_atomic_factor_projection() -> None:
    relation = _relation(status="PARTIAL")
    factorization = factorize_relational_atomic_identity(
        relation=relation,
        candidate=_candidate(),
        candidate_hypothesis_id="hypothesis:candidate",
    )
    projections = compile_relational_atomic_factor_projection_set(
        relation=relation,
        factorization=factorization,
    )

    assert factorization.factorization_status == "RELATION_NOT_READY"
    assert projections.planning_ready is False
    assert projections.projection_count == 0
