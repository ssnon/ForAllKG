from __future__ import annotations

import copy
import inspect

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    compile_epistemic_modality_fidelity_shadow,
    plan_modal_preserving_claim_text_repair_shadow,
    preview_modal_preserving_claim_text_repair_shadow,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
)


def _claim(
    *,
    text: str = "The factor moderates the outcome.",
    basis: str = "The factor may moderate the outcome.",
    bridge: str = "The factor can moderate the outcome.",
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


def _semantic(
    *,
    basis: str = "The factor may moderate the outcome.",
) -> dict:
    return {
        "proposition_basis": basis,
        "proposition_basis_source_contexts": [
            {
                "source_path": "inferential_bridge",
                "sentence": basis,
            }
        ],
        "reason_codes": [],
    }


def _modal_shadow(
    claim: NoveltyClaimDraft,
    *,
    semantic: dict | None = None,
) -> dict:
    return compile_epistemic_modality_fidelity_shadow(
        claim,
        semantic_fidelity_shadow=(
            semantic or _semantic(
                basis=claim.semantic_fidelity_binding.proposition_basis
            )
        ),
        sanitized_required_bridge=claim.required_bridge,
    )


def test_planner_and_preview_restore_bound_source_modal_text_only() -> None:
    claim = _claim()
    semantic = _semantic()
    claim_before = claim.model_dump(mode="json")
    semantic_before = copy.deepcopy(semantic)

    modal_shadow = _modal_shadow(
        claim,
        semantic=semantic,
    )
    plan = plan_modal_preserving_claim_text_repair_shadow(
        claim,
        epistemic_modality_fidelity_shadow=modal_shadow,
    )
    preview = preview_modal_preserving_claim_text_repair_shadow(
        claim,
        semantic_fidelity_shadow=semantic,
        sanitized_required_bridge=claim.required_bridge,
        repair_plan_shadow=plan,
    )

    assert plan["status"] == (
        "MODAL_PRESERVING_TEXT_REPAIR_CANDIDATE_SHADOW"
    )
    assert plan["source_authority"] == "bound_proposition_basis"
    assert plan["required_bridge_is_authority"] is False
    assert plan["source_modal"] == "may"
    assert plan["replacement_surface"] == "may moderate"
    assert plan["production_authority"] is False

    assert preview["status"] == "PREVIEW_READY_SHADOW"
    assert preview["before_text"] == (
        "The factor moderates the outcome."
    )
    assert preview["preview_text"] == (
        "The factor may moderate the outcome."
    )
    assert preview["changed_field_names"] == ["text"]
    assert preview["modal_strengthening_candidate_cleared"] is True
    assert (
        preview["post_preview_modality_shadow"]["shadow_status"]
        == "NO_MODAL_STRENGTHENING_CANDIDATE"
    )
    assert preview["production_authority"] is False
    assert preview["canonical_mutation_performed"] is False
    assert claim.model_dump(mode="json") == claim_before
    assert semantic == semantic_before


def test_required_bridge_modal_is_not_repair_authority() -> None:
    claim = _claim(
        bridge="The factor could moderate the outcome.",
    )
    semantic = _semantic()
    shadow = _modal_shadow(
        claim,
        semantic=semantic,
    )

    plan = plan_modal_preserving_claim_text_repair_shadow(
        claim,
        epistemic_modality_fidelity_shadow=shadow,
    )

    assert plan["source_modal"] == "may"
    assert plan["replacement_surface"] == "may moderate"
    assert plan["required_bridge_is_authority"] is False
    assert plan["modal_token_equivalence_assumed"] is False


def test_multiple_source_modal_mentions_fail_closed() -> None:
    claim = _claim()
    shadow = copy.deepcopy(
        _modal_shadow(claim)
    )
    candidate = shadow["modal_drop_candidates"][0]
    candidate["source_basis_modal_mentions"] = [
        *candidate["source_basis_modal_mentions"],
        {
            "modal": "could",
            "predicate_lemma": "moderate",
            "predicate_surface": "moderate",
            "surface": "could moderate",
        },
    ]

    plan = plan_modal_preserving_claim_text_repair_shadow(
        claim,
        epistemic_modality_fidelity_shadow=shadow,
    )

    assert plan["status"] == (
        "NO_REPAIR_SOURCE_MODAL_CARDINALITY"
    )
    assert plan["production_authority"] is False


def test_multiple_claim_predicate_mentions_fail_closed() -> None:
    claim = _claim(
        text=(
            "The factor moderates the outcome and "
            "moderates the response."
        ),
    )
    shadow = _modal_shadow(claim)

    plan = plan_modal_preserving_claim_text_repair_shadow(
        claim,
        epistemic_modality_fidelity_shadow=shadow,
    )

    assert plan["status"] in {
        "NO_REPAIR_CLAIM_PREDICATE_CARDINALITY",
        "NO_REPAIR_CLAIM_TARGET_NOT_EXACT_UNIQUE",
    }
    assert plan["production_authority"] is False


def test_noncandidate_does_not_plan_or_preview() -> None:
    claim = _claim(
        text="The factor may moderate the outcome.",
    )
    semantic = _semantic()
    shadow = _modal_shadow(
        claim,
        semantic=semantic,
    )

    plan = plan_modal_preserving_claim_text_repair_shadow(
        claim,
        epistemic_modality_fidelity_shadow=shadow,
    )
    preview = preview_modal_preserving_claim_text_repair_shadow(
        claim,
        semantic_fidelity_shadow=semantic,
        sanitized_required_bridge=claim.required_bridge,
        repair_plan_shadow=plan,
    )

    assert plan["status"] == (
        "NO_REPAIR_NOT_MODAL_STRENGTHENING_CANDIDATE"
    )
    assert preview["status"] == "NO_PREVIEW_PLAN_NOT_READY"


def test_stale_plan_is_blocked_without_mutation() -> None:
    claim = _claim()
    semantic = _semantic()
    shadow = _modal_shadow(
        claim,
        semantic=semantic,
    )
    plan = plan_modal_preserving_claim_text_repair_shadow(
        claim,
        epistemic_modality_fidelity_shadow=shadow,
    )
    stale = copy.deepcopy(plan)
    stale["claim_target_surface"] = "promotes"

    before = claim.model_dump(mode="json")
    preview = preview_modal_preserving_claim_text_repair_shadow(
        claim,
        semantic_fidelity_shadow=semantic,
        sanitized_required_bridge=claim.required_bridge,
        repair_plan_shadow=stale,
    )

    assert preview["status"] == "NO_PREVIEW_STALE_OR_INVALID_PLAN"
    assert claim.model_dump(mode="json") == before


def test_decomposer_records_plan_and_preview_as_diagnostics_only() -> None:
    source = inspect.getsource(NoveltyClaimDecomposer.decompose)

    assert "plan_modal_preserving_claim_text_repair_shadow" in source
    assert "preview_modal_preserving_claim_text_repair_shadow" in source
    assert '"modal_preserving_claim_text_repair_plan_shadow"' in source
    assert '"modal_preserving_claim_text_repair_preview_shadow"' in source

    init_source = inspect.getsource(NoveltyClaimDecomposer.__init__)
    assert "modal_preserving_claim_text_repair_plan_shadow" not in init_source
    assert "modal_preserving_claim_text_repair_preview_shadow" not in init_source

def test_preview_preserves_non_target_unicode_punctuation_byte_exact() -> None:
    before_text = (
        "At comparable overall metal–metal electronic coupling, "
        "the bonding–antibonding distribution moderates "
        "hydrogen adsorption–desorption compatibility."
    )
    basis = (
        "At comparable overall metal-metal electronic coupling, "
        "the bonding-antibonding distribution may moderate "
        "hydrogen adsorption-desorption compatibility."
    )
    bridge = (
        "At comparable overall metal–metal electronic coupling, "
        "the bonding–antibonding distribution can moderate "
        "hydrogen adsorption–desorption compatibility."
    )
    claim = _claim(
        text=before_text,
        basis=basis,
        bridge=bridge,
    )
    semantic = _semantic(basis=basis)
    # _claim(..., bridge=bridge) already stores this exact bridge on
    # claim.required_bridge; the existing _modal_shadow helper intentionally
    # accepts only claim + optional semantic diagnostics.
    shadow = _modal_shadow(
        claim,
        semantic=semantic,
    )

    plan = plan_modal_preserving_claim_text_repair_shadow(
        claim,
        epistemic_modality_fidelity_shadow=shadow,
    )
    preview = preview_modal_preserving_claim_text_repair_shadow(
        claim,
        semantic_fidelity_shadow=semantic,
        sanitized_required_bridge=bridge,
        repair_plan_shadow=plan,
    )

    expected = (
        "At comparable overall metal–metal electronic coupling, "
        "the bonding–antibonding distribution may moderate "
        "hydrogen adsorption–desorption compatibility."
    )
    assert plan["status"] == (
        "MODAL_PRESERVING_TEXT_REPAIR_CANDIDATE_SHADOW"
    )
    assert preview["status"] == "PREVIEW_READY_SHADOW"
    assert preview["before_text"] == before_text
    assert preview["preview_text"] == expected
    assert "metal–metal" in preview["preview_text"]
    assert "bonding–antibonding" in preview["preview_text"]
    assert "adsorption–desorption" in preview["preview_text"]
    assert "metal-metal" not in preview["preview_text"]
    assert preview["changed_field_names"] == ["text"]
    assert preview["modal_strengthening_candidate_cleared"] is True

