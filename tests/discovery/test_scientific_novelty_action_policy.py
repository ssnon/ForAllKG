from __future__ import annotations

from pipeline_core.discovery.scientific_novelty_action_policy import (
    ScientificNoveltyActionPolicy,
)


def _policy():
    return ScientificNoveltyActionPolicy()


def test_t01_retrospective_fixture_is_keep_eligible():
    decision = _policy().evaluate(
        external_status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        semantic_pass_1="MODERATE",
        semantic_pass_2="MODERATE",
    )

    assert decision.semantic_stable is True
    assert (
        decision.stable_semantic_tier
        == "MODERATE"
    )
    assert decision.action == "KEEP_ELIGIBLE"
    assert decision.selection_class == "ELIGIBLE"
    assert decision.shadow_only is True
    assert (
        decision.production_selection_changed
        is False
    )


def test_t03_slot1_stable_low_requires_reaxis():
    decision = _policy().evaluate(
        external_status=(
            "LITERATURE_SUPPORTED_EXTENSION"
        ),
        semantic_pass_1="LOW",
        semantic_pass_2="LOW",
    )

    assert decision.semantic_stable is True
    assert decision.stable_semantic_tier == "LOW"
    assert decision.action == "REAXIS_REQUIRED"
    assert (
        decision.selection_class
        == "INELIGIBLE"
    )
    assert (
        "STABLE_SEMANTIC_LOW"
        in decision.reason_codes
    )


def test_t03_slot2_unstable_is_unresolved():
    decision = _policy().evaluate(
        external_status=(
            "LITERATURE_SUPPORTED_EXTENSION"
        ),
        semantic_pass_1="MODERATE",
        semantic_pass_2="LOW",
    )

    assert decision.semantic_stable is False
    assert decision.stable_semantic_tier is None
    assert decision.action == "UNRESOLVED"
    assert (
        decision.selection_class
        == "INELIGIBLE"
    )


def test_well_established_is_ineligible_even_if_high():
    decision = _policy().evaluate(
        external_status="WELL_ESTABLISHED",
        semantic_pass_1="HIGH",
        semantic_pass_2="HIGH",
    )

    assert decision.action == "REAXIS_REQUIRED"
    assert (
        decision.selection_class
        == "INELIGIBLE"
    )
    assert (
        "DESTRUCTIVE_EXTERNAL_STATUS"
        in decision.reason_codes
    )


def test_conflicting_prior_art_is_ineligible():
    decision = _policy().evaluate(
        external_status="CONFLICTING_PRIOR_ART",
        semantic_pass_1="MODERATE",
        semantic_pass_2="MODERATE",
    )

    assert decision.action == "REAXIS_REQUIRED"
    assert (
        decision.selection_class
        == "INELIGIBLE"
    )


def test_indeterminate_semantic_requires_evidence():
    decision = _policy().evaluate(
        external_status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        semantic_pass_1="INDETERMINATE",
        semantic_pass_2="INDETERMINATE",
    )

    assert decision.action == "EVIDENCE_REQUIRED"
    assert (
        decision.selection_class
        == "INELIGIBLE"
    )


def test_insufficient_external_search_requires_evidence():
    decision = _policy().evaluate(
        external_status=(
            "INSUFFICIENT_SEARCH_EVIDENCE"
        ),
        semantic_pass_1="HIGH",
        semantic_pass_2="HIGH",
    )

    assert decision.action == "EVIDENCE_REQUIRED"
    assert (
        decision.selection_class
        == "INELIGIBLE"
    )


def test_literature_extension_moderate_is_conditional():
    decision = _policy().evaluate(
        external_status=(
            "LITERATURE_SUPPORTED_EXTENSION"
        ),
        semantic_pass_1="MODERATE",
        semantic_pass_2="MODERATE",
    )

    assert decision.action == "REFINE_OR_REAXIS"
    assert (
        decision.selection_class
        == "CONDITIONAL"
    )


def test_relational_gap_high_is_keep_eligible():
    decision = _policy().evaluate(
        external_status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        semantic_pass_1="HIGH",
        semantic_pass_2="HIGH",
    )

    assert decision.action == "KEEP_ELIGIBLE"
    assert decision.selection_class == "ELIGIBLE"


def test_plausibly_novel_low_is_still_reaxis_required():
    decision = _policy().evaluate(
        external_status="PLAUSIBLY_NOVEL",
        semantic_pass_1="LOW",
        semantic_pass_2="LOW",
    )

    # Exact relation novelty cannot override obvious/reconstructable
    # scientific structure.
    assert decision.action == "REAXIS_REQUIRED"
    assert (
        decision.selection_class
        == "INELIGIBLE"
    )


def test_new_combination_moderate_is_keep_eligible():
    decision = _policy().evaluate(
        external_status=(
            "NEW_COMBINATION_OF_KNOWN_EFFECTS"
        ),
        semantic_pass_1="MODERATE",
        semantic_pass_2="MODERATE",
    )

    assert decision.action == "KEEP_ELIGIBLE"
    assert decision.selection_class == "ELIGIBLE"


def test_shadow_contract_never_claims_production_mutation():
    decision = _policy().evaluate(
        external_status="PLAUSIBLY_NOVEL",
        semantic_pass_1="HIGH",
        semantic_pass_2="HIGH",
    )

    assert decision.shadow_only is True
    assert decision.action_policy_applied is False
    assert (
        decision.scientific_selection_changed
        is False
    )
    assert (
        decision.production_selection_changed
        is False
    )
