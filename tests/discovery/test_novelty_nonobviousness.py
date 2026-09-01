from pipeline_core.discovery.novelty_nonobviousness import (
    NonObviousnessEvidenceClosure,
    ResidualClaimStructure,
    assess_structural_nonobviousness,
)


def test_missing_power_lower_order_relations_is_insufficient_closure():
    result = assess_structural_nonobviousness(
        NonObviousnessEvidenceClosure(
            base_relation="ESTABLISHED",
            distinguishing_factor_effect="NOT_FOUND",
            bridge_relation="NOT_FOUND",
            full_relation="NOT_FOUND",
            bridge_kind="NONE",
        ),
        ResidualClaimStructure(
            claim_kind="moderator_interaction",
        ),
    )

    assert result.status == "INSUFFICIENT_CLOSURE"


def test_separate_main_effects_do_not_make_moderation_routine():
    result = assess_structural_nonobviousness(
        NonObviousnessEvidenceClosure(
            base_relation="ESTABLISHED",
            distinguishing_factor_effect="ESTABLISHED",
            bridge_relation="ESTABLISHED",
            full_relation="NOT_FOUND",
            bridge_kind="MAIN_EFFECTS_ONLY",
        ),
        ResidualClaimStructure(
            claim_kind="moderator_interaction",
        ),
    )

    assert result.status == "INTERACTION_LEAP"


def test_mediation_chain_does_not_make_moderation_routine():
    result = assess_structural_nonobviousness(
        NonObviousnessEvidenceClosure(
            base_relation="ESTABLISHED",
            distinguishing_factor_effect="ESTABLISHED",
            bridge_relation="ESTABLISHED",
            full_relation="NOT_FOUND",
            bridge_kind="MEDIATION_CHAIN",
        ),
        ResidualClaimStructure(
            claim_kind="moderator_interaction",
        ),
    )

    assert result.status == "INTERACTION_LEAP"


def test_interaction_compatible_bridge_can_be_routine():
    result = assess_structural_nonobviousness(
        NonObviousnessEvidenceClosure(
            base_relation="ESTABLISHED",
            distinguishing_factor_effect="ESTABLISHED",
            bridge_relation="ESTABLISHED",
            full_relation="NOT_FOUND",
            bridge_kind="INTERACTION_COMPATIBLE",
        ),
        ResidualClaimStructure(
            claim_kind="moderator_interaction",
        ),
    )

    assert result.status == "ROUTINE_COMPOSITION"


def test_threshold_is_not_reduced_to_known_components():
    result = assess_structural_nonobviousness(
        NonObviousnessEvidenceClosure(
            base_relation="ESTABLISHED",
            distinguishing_factor_effect="ESTABLISHED",
            bridge_relation="ESTABLISHED",
            full_relation="NOT_FOUND",
            bridge_kind="MEDIATION_CHAIN",
        ),
        ResidualClaimStructure(
            claim_kind="distinctive_prediction",
            introduces_threshold=True,
        ),
    )

    assert result.status == "REGIME_OR_THRESHOLD_LEAP"


def test_direct_prior_art_dominates():
    result = assess_structural_nonobviousness(
        NonObviousnessEvidenceClosure(
            base_relation="ESTABLISHED",
            distinguishing_factor_effect="ESTABLISHED",
            bridge_relation="ESTABLISHED",
            full_relation="ESTABLISHED",
            bridge_kind="INTERACTION_COMPATIBLE",
        ),
        ResidualClaimStructure(
            claim_kind="moderator_interaction",
        ),
    )

    assert result.status == "DIRECTLY_KNOWN"
