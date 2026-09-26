from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    NoveltyClaim,
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)
from pipeline_core.discovery.legacy_atomic_specification_shadow import (
    compile_legacy_atomic_specification_shadow_report,
)


def _card(
    *,
    statement: str = (
        "Under condition C, Factor M moderates the relationship between "
        "descriptor D and outcome O."
    ),
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="hypothesis:h1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Factor M and outcome O",
        hypothesis_statement=statement,
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        inferential_bridge=statement,
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:p1",
                observable="Outcome O under condition C",
                expected_direction="qualitative_change",
                rationale=(
                    "Outcome O should differ when Factor M changes "
                    "under condition C."
                ),
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:f1",
                observable="Outcome O under condition C",
                falsifying_outcome=(
                    "Outcome O does not differ when Factor M changes "
                    "under condition C."
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


def _draft_claim(
    *,
    text: str | None = None,
    basis: str | None = None,
    bridge: str | None = None,
    direction: list[str] | None = None,
    prediction_id: str | None = "prediction:p1",
    falsifier_id: str | None = "falsifier:f1",
) -> NoveltyClaimDraft:
    source = (
        "Under condition C, Factor M moderates the relationship between "
        "descriptor D and outcome O."
    )
    return NoveltyClaimDraft(
        local_id="c1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=text or source,
        rationale="Atomic claim for shadow compilation.",
        search_concepts=["Factor M", "descriptor D", "outcome O"],
        search_queries=["Factor M descriptor D outcome O"],
        distinguishing_terms=["condition C"],
        prior_art_identity_terms=["Factor M"],
        relation_nucleus_terms=["descriptor D", "outcome O", "moderates"],
        required_bridge=bridge if bridge is not None else source,
        predicted_observation=(
            "A qualitative change in Outcome O is expected when "
            "Factor M changes under condition C."
        ),
        falsification_condition=(
            "No qualitative change in Outcome O is observed when "
            "Factor M changes under condition C."
        ),
        semantic_fidelity_binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis=basis if basis is not None else source,
            relation_endpoint_anchors=[
                "Factor M",
                "descriptor D",
                "outcome O",
            ],
            scope_qualifier_spans=["Under condition C"],
            directional_qualifier_spans=direction or [],
            prediction_observation_id=prediction_id,
            falsification_criterion_id=falsifier_id,
        ),
    )


def _canonical(
    draft: NoveltyClaimDraft,
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="external_novelty_claim:c1",
        hypothesis_id="hypothesis:h1",
        claim_rank=1,
        kind=draft.kind,
        importance=draft.importance,
        novelty_selection_role=draft.novelty_selection_role,
        text=draft.text,
        rationale=draft.rationale,
        search_concepts=list(draft.search_concepts),
        search_queries=list(draft.search_queries),
        distinguishing_terms=list(draft.distinguishing_terms),
        prior_art_identity_terms=list(draft.prior_art_identity_terms),
        relation_nucleus_terms=list(draft.relation_nucleus_terms),
        required_bridge=draft.required_bridge,
        predicted_observation=draft.predicted_observation,
        falsification_condition=draft.falsification_condition,
    )


def _compile(
    card: HypothesisCard,
    draft_claim: NoveltyClaimDraft,
):
    canonical = _canonical(draft_claim)
    return compile_legacy_atomic_specification_shadow_report(
        hypothesis=card,
        decomposition_draft=NoveltyClaimDecompositionDraft(
            claims=[draft_claim]
        ),
        canonical_claims=HypothesisNoveltyClaims(
            hypothesis_id=card.hypothesis_id,
            title=card.title,
            claims=[canonical],
        ),
    )


def test_shadow_preserves_source_identity_without_mutating_canonical_surfaces():
    card = _card()
    draft = _draft_claim()
    report = _compile(card, draft)

    assert report.diagnostic_only is True
    assert report.production_authority is False
    assert report.vpre_contract_changed is False
    assert report.production_selection_changed is False
    assert report.row_count == 1
    assert report.compiled_count == 1

    row = report.rows[0]
    assert row.source_reference_status == "READY"
    assert row.proposition_fidelity_status == "PASS"
    assert row.bridge_fidelity_status == "PASS"
    assert row.semantic_fidelity_status == "PASS"
    assert row.specification_status == "COMPLETE"
    assert row.atomic_kind_status == "SUPPORTED"
    assert row.compilation_status == "COMPILED_SHADOW"

    spec = row.specification
    assert spec is not None
    assert spec.prediction_observation_id == "prediction:p1"
    assert spec.falsification_criterion_id == "falsifier:f1"
    assert spec.observable == "Outcome O under condition C"

    # The shadow representation preserves canonical claim surfaces verbatim;
    # source IDs/observable are added as provenance, not used to rewrite text.
    assert spec.predicted_observation == draft.predicted_observation
    assert spec.falsification_condition == draft.falsification_condition
    assert spec.required_bridge == draft.required_bridge
    assert row.canonical_claim_mutated is False


def test_shadow_separates_source_ready_from_semantic_invalid():
    source = (
        "Under condition C, increasing Factor M increases outcome O "
        "relative to descriptor D."
    )
    card = _card(statement=source)
    draft = _draft_claim(
        text=(
            "Under condition C, Factor M is associated with outcome O "
            "relative to descriptor D."
        ),
        basis=source,
        bridge=(
            "Under condition C, Factor M is associated with outcome O "
            "relative to descriptor D."
        ),
        direction=["increases"],
    )

    report = _compile(card, draft)
    row = report.rows[0]

    assert row.source_reference_status == "READY"
    assert row.proposition_fidelity_status == "INVALID"
    assert row.semantic_fidelity_status == "INVALID"
    assert (
        "atomic_claim_direction_qualifier_not_preserved"
        in row.proposition_fidelity_reason_codes
    )

    # A structurally compiled shadow object may still exist, but it has no
    # production authority and does not imply semantic admissibility.
    assert row.compilation_status == "COMPILED_SHADOW"
    assert row.specification is not None
    assert row.production_authority is False


def test_shadow_invalid_source_reference_abstains_compilation():
    card = _card()
    draft = _draft_claim(
        prediction_id="prediction:unknown",
    )

    report = _compile(card, draft)
    row = report.rows[0]

    assert row.source_reference_status == "INVALID"
    assert row.compilation_status == "ABSTAINED_SOURCE_REFERENCE"
    assert row.specification is None
    assert row.source_reference_reason_codes == [
        "prediction_source_id_cardinality:0"
    ]
    assert "atomic_prediction_source_id_unknown" in (
        row.semantic_fidelity_reason_codes
    )
    assert row.production_authority is False


def test_shadow_keeps_specification_completeness_independent_from_source_binding():
    card = _card()
    draft = _draft_claim()
    canonical = _canonical(draft).model_copy(
        update={"required_bridge": ""}
    )

    report = compile_legacy_atomic_specification_shadow_report(
        hypothesis=card,
        decomposition_draft=NoveltyClaimDecompositionDraft(
            claims=[draft]
        ),
        canonical_claims=HypothesisNoveltyClaims(
            hypothesis_id=card.hypothesis_id,
            title=card.title,
            claims=[canonical],
        ),
    )
    row = report.rows[0]

    assert row.source_reference_status == "READY"
    assert row.specification_status == "INCOMPLETE"
    assert row.specification_reason_codes == [
        "missing_required_bridge"
    ]
    assert row.compilation_status == "COMPILED_SHADOW"
    assert row.specification is not None
    assert row.specification.required_bridge == ""
    assert row.vpre_contract_changed is False
