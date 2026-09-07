from __future__ import annotations

import copy
import inspect

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    compile_epistemic_modality_fidelity_shadow,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
)


def _claim(
    *,
    text: str,
    basis: str,
    bridge: str,
) -> NoveltyClaimDraft:
    return NoveltyClaimDraft(
        local_id="claim_1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=text,
        rationale="diagnostic test",
        search_concepts=["factor", "outcome"],
        search_queries=["factor outcome relation"],
        prior_art_identity_terms=["factor"],
        relation_nucleus_terms=["factor", "outcome"],
        required_bridge=bridge,
        predicted_observation="",
        falsification_condition="",
        semantic_fidelity_binding=(
            NoveltyClaimSemanticFidelityBindingDraft(
                proposition_basis=basis,
                relation_endpoint_anchors=[
                    "factor",
                    "outcome",
                ],
                scope_qualifier_spans=[],
                directional_qualifier_spans=[],
                prediction_observation_id=None,
                falsification_criterion_id=None,
            )
        ),
    )


def _review(
    *,
    basis: str,
    context: str | None = None,
) -> dict:
    return {
        "proposition_basis": basis,
        "proposition_basis_source_contexts": [
            {
                "source_path": "inferential_bridge",
                "sentence": context or basis,
            }
        ],
        "reason_codes": [
            "atomic_claim_context_comparative_scope_may_be_broadened",
        ],
    }


def test_shadow_flags_exact_lexical_modal_drop_without_authority() -> None:
    basis = "The factor may moderate the outcome."
    bridge = "The factor can moderate the outcome."
    claim = _claim(
        text="The factor moderates the outcome.",
        basis=basis,
        bridge=bridge,
    )
    review = _review(basis=basis)

    review_before = copy.deepcopy(review)
    claim_before = claim.model_dump(mode="json")

    shadow = compile_epistemic_modality_fidelity_shadow(
        claim,
        semantic_fidelity_shadow=review,
        sanitized_required_bridge=bridge,
    )

    assert shadow["diagnostic_only"] is True
    assert shadow["production_authority"] is False
    assert shadow["reason_codes_modified"] is False
    assert shadow["taxonomy_modified"] is False
    assert shadow["semantic_synonymy_allowed"] is False
    assert shadow["modal_token_equivalence_assumed"] is False
    assert shadow["shadow_status"] == (
        "MODAL_STRENGTHENING_CANDIDATE"
    )
    assert shadow["candidate_reason_code"] == (
        "atomic_claim_epistemic_modality_may_be_strengthened"
    )
    assert len(shadow["modal_drop_candidates"]) == 1
    candidate = shadow["modal_drop_candidates"][0]
    assert candidate["predicate_lemma"] == "moderate"
    assert candidate["bridge_modal_preservation_observed"] is True
    assert shadow["current_modal_reason_codes"] == []
    assert review == review_before
    assert claim.model_dump(mode="json") == claim_before


def test_same_modalized_claim_is_not_a_drop_candidate() -> None:
    basis = "The factor may moderate the outcome."
    bridge = "The factor can moderate the outcome."
    claim = _claim(
        text="The factor may moderate the outcome.",
        basis=basis,
        bridge=bridge,
    )

    shadow = compile_epistemic_modality_fidelity_shadow(
        claim,
        semantic_fidelity_shadow=_review(basis=basis),
        sanitized_required_bridge=bridge,
    )

    assert shadow["modal_drop_candidates"] == []
    assert shadow["candidate_reason_code"] is None
    assert shadow["shadow_status"] == (
        "NO_MODAL_STRENGTHENING_CANDIDATE"
    )


def test_modal_rewording_without_drop_is_not_flagged() -> None:
    basis = "The factor may moderate the outcome."
    bridge = "The factor can moderate the outcome."
    claim = _claim(
        text="The factor could moderate the outcome.",
        basis=basis,
        bridge=bridge,
    )

    shadow = compile_epistemic_modality_fidelity_shadow(
        claim,
        semantic_fidelity_shadow=_review(basis=basis),
        sanitized_required_bridge=bridge,
    )

    assert shadow["modal_drop_candidates"] == []
    assert {
        row["modal"]
        for row in shadow["claim_modal_mentions"]
    } == {"could"}
    assert shadow["modal_token_equivalence_assumed"] is False


def test_different_predicate_is_not_treated_as_synonymous_support() -> None:
    basis = "The factor may moderate the outcome."
    bridge = "The factor may moderate the outcome."
    claim = _claim(
        text="The factor promotes the outcome.",
        basis=basis,
        bridge=bridge,
    )

    shadow = compile_epistemic_modality_fidelity_shadow(
        claim,
        semantic_fidelity_shadow=_review(basis=basis),
        sanitized_required_bridge=bridge,
    )

    assert shadow["modal_drop_candidates"] == []
    assert shadow["semantic_synonymy_allowed"] is False


def test_context_only_modal_cannot_launder_unmodalized_basis() -> None:
    basis = "The factor moderates the outcome."
    context = (
        "The factor moderates the outcome; "
        "a sibling mechanism may support another endpoint."
    )
    claim = _claim(
        text="The factor moderates the outcome.",
        basis=basis,
        bridge=basis,
    )

    shadow = compile_epistemic_modality_fidelity_shadow(
        claim,
        semantic_fidelity_shadow=_review(
            basis=basis,
            context=context,
        ),
        sanitized_required_bridge=basis,
    )

    assert shadow["source_basis_modal_mentions"] == []
    assert shadow["source_context_modal_mentions"] != []
    assert shadow["modal_drop_candidates"] == []


def test_negated_predicate_is_not_classified_as_modal_strengthening() -> None:
    basis = "The factor may moderate the outcome."
    bridge = "The factor may moderate the outcome."
    claim = _claim(
        text="The factor does not moderate the outcome.",
        basis=basis,
        bridge=bridge,
    )

    shadow = compile_epistemic_modality_fidelity_shadow(
        claim,
        semantic_fidelity_shadow=_review(basis=basis),
        sanitized_required_bridge=bridge,
    )

    assert shadow["modal_drop_candidates"] == []


def test_decomposer_records_modality_shadow_only_in_diagnostic_record() -> None:
    source = inspect.getsource(NoveltyClaimDecomposer.decompose)

    assert "compile_epistemic_modality_fidelity_shadow" in source
    assert '"epistemic_modality_fidelity_shadow"' in source
    assert "epistemic_modality_fidelity_shadow" not in inspect.getsource(
        NoveltyClaimDecomposer.__init__
    )
