from pipeline_core.discovery.novelty_adjudication import (
    NonObviousnessAdjudicationVector,
    assess_adjudication_readiness,
)


def vector(
    *,
    distance="NEW_RELATIONAL_FORM",
    mechanism="NEW_BRIDGE_REQUIRED",
    regime="CONDITIONED",
    counter="NONTRIVIAL",
    testability="COMPARATIVE",
    bridge="A new bridge is required.",
    prediction="Condition A differs from condition B.",
    falsifier="No condition-dependent difference is observed.",
):
    return NonObviousnessAdjudicationVector(
        inferential_distance=distance,
        mechanistic_necessity=mechanism,
        regime_specificity=regime,
        counterintuitiveness=counter,
        testable_distinctiveness=testability,
        required_bridge=bridge,
        predicted_observation=prediction,
        falsification_condition=falsifier,
    )


def test_insufficient_closure_is_not_eligible():
    result = assess_adjudication_readiness(
        structural_status="INSUFFICIENT_CLOSURE",
        vector=vector(),
    )

    assert result.readiness == "NOT_ELIGIBLE"


def test_generic_interaction_needs_refinement():
    result = assess_adjudication_readiness(
        structural_status="INTERACTION_LEAP",
        vector=vector(
            distance="SINGLE_KNOWN_STEP",
            mechanism="KNOWN_MECHANISM_REUSED",
            regime="CONDITIONED",
            testability="GENERIC",
        ),
    )

    assert result.readiness == "NEEDS_REFINEMENT"


def test_interaction_with_explicit_new_bridge_can_be_reviewed():
    result = assess_adjudication_readiness(
        structural_status="INTERACTION_LEAP",
        vector=vector(
            distance="NEW_RELATIONAL_FORM",
            mechanism="NEW_BRIDGE_REQUIRED",
            regime="CONDITIONED",
            testability="COMPARATIVE",
        ),
    )

    assert (
        result.readiness
        == "READY_FOR_NONOBVIOUSNESS_REVIEW"
    )


def test_threshold_control_is_review_ready():
    result = assess_adjudication_readiness(
        structural_status="REGIME_OR_THRESHOLD_LEAP",
        vector=vector(
            distance="NEW_REGIME_STRUCTURE",
            mechanism="NEW_BRIDGE_REQUIRED",
            regime="THRESHOLD",
            counter="NONTRIVIAL",
            testability="QUANTITATIVE",
            bridge=(
                "A power-driven transition changes the "
                "spacing-to-SERS response."
            ),
            prediction=(
                "The response changes discontinuously near "
                "a critical laser power."
            ),
            falsifier=(
                "The response varies smoothly with power and "
                "shows no critical regime."
            ),
        ),
    )

    assert (
        result.readiness
        == "READY_FOR_NONOBVIOUSNESS_REVIEW"
    )


def test_fake_threshold_without_threshold_structure_is_rejected():
    result = assess_adjudication_readiness(
        structural_status="REGIME_OR_THRESHOLD_LEAP",
        vector=vector(
            regime="CONDITIONED",
        ),
    )

    assert result.readiness == "NEEDS_REFINEMENT"


def test_mechanistic_leap_requires_new_mechanism():
    result = assess_adjudication_readiness(
        structural_status="MECHANISTIC_LEAP",
        vector=vector(
            mechanism="KNOWN_MECHANISM_REUSED",
        ),
    )

    assert result.readiness == "NEEDS_REFINEMENT"


def test_missing_falsifier_needs_refinement():
    result = assess_adjudication_readiness(
        structural_status="INTERACTION_LEAP",
        vector=vector(
            falsifier="",
        ),
    )

    assert result.readiness == "NEEDS_REFINEMENT"



# N9-C2b compiler tests

from pipeline_core.discovery.novelty_adjudication import (
    EstablishedPriorArtRelation,
    NonObviousnessAdjudicationDraft,
    NonObviousnessEvidencePacket,
    NonObviousnessReviewGate,
    compile_nonobviousness_adjudication,
)


def ready_gate():
    return NonObviousnessReviewGate(
        readiness="READY_FOR_NONOBVIOUSNESS_REVIEW",
        reason_codes=("test",),
        interpretation="test",
    )


def evidence_packet(
    *,
    direct=False,
    closure=True,
):
    return NonObviousnessEvidencePacket(
        claim_id="claim:test",
        claim_text=(
            "A critical power separates two spacing-to-SERS regimes."
        ),
        structural_status="REGIME_OR_THRESHOLD_LEAP",
        vector=vector(
            distance="NEW_REGIME_STRUCTURE",
            mechanism="NEW_BRIDGE_REQUIRED",
            regime="THRESHOLD",
            testability="QUANTITATIVE",
        ),
        established_relations=(
            EstablishedPriorArtRelation(
                relation_statement="Spacing affects SERS.",
                relationship_status="DIRECT_PRIOR_ART",
                work_ids=("work:1",),
            ),
        ),
        direct_full_claim_prior_art=direct,
        evidence_closure_sufficient=closure,
    )


def test_direct_full_prior_art_forces_routine():
    result = compile_nonobviousness_adjudication(
        readiness=ready_gate(),
        packet=evidence_packet(direct=True),
        draft=NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(
                "A new transition exists.",
            ),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis="test",
        ),
    )

    assert result.verdict == "ROUTINE_FROM_PRIOR_ART"


def test_incomplete_closure_cannot_be_called_nonobvious():
    result = compile_nonobviousness_adjudication(
        readiness=ready_gate(),
        packet=evidence_packet(closure=False),
        draft=NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(
                "A new transition exists.",
            ),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis="test",
        ),
    )

    assert result.verdict == "INSUFFICIENT_FOR_JUDGMENT"


def test_direct_reconstruction_forces_routine():
    result = compile_nonobviousness_adjudication(
        readiness=ready_gate(),
        packet=evidence_packet(),
        draft=NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=True,
            additional_scientific_assumptions=(
                "claimed but unnecessary",
            ),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis="test",
        ),
    )

    assert result.verdict == "ROUTINE_FROM_PRIOR_ART"


def test_missing_additional_bridge_cannot_be_nonobvious():
    result = compile_nonobviousness_adjudication(
        readiness=ready_gate(),
        packet=evidence_packet(),
        draft=NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis="test",
        ),
    )

    assert result.verdict == "INSUFFICIENT_FOR_JUDGMENT"


def test_generic_prediction_cannot_be_nonobvious():
    result = compile_nonobviousness_adjudication(
        readiness=ready_gate(),
        packet=evidence_packet(),
        draft=NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(
                "A power-driven transition changes the response regime.",
            ),
            prediction_distinguishes_from_routine_baseline=False,
            falsifier_is_specific=True,
            concise_basis="test",
        ),
    )

    assert result.verdict == "INSUFFICIENT_FOR_JUDGMENT"


def test_specific_threshold_candidate_can_be_potentially_nonobvious():
    result = compile_nonobviousness_adjudication(
        readiness=ready_gate(),
        packet=evidence_packet(),
        draft=NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(
                "Laser power drives a state transition that changes "
                "how spacing maps to SERS.",
            ),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis=(
                "Known spacing and optical-response relations do not "
                "supply the proposed critical transition."
            ),
        ),
    )

    assert result.verdict == "POTENTIALLY_NON_OBVIOUS"


def test_not_ready_candidate_is_never_adjudicated_positive():
    result = compile_nonobviousness_adjudication(
        readiness=NonObviousnessReviewGate(
            readiness="NEEDS_REFINEMENT",
            reason_codes=("generic",),
            interpretation="generic",
        ),
        packet=evidence_packet(),
        draft=NonObviousnessAdjudicationDraft(
            proposed_verdict="POTENTIALLY_NON_OBVIOUS",
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(
                "new bridge",
            ),
            prediction_distinguishes_from_routine_baseline=True,
            falsifier_is_specific=True,
            concise_basis="test",
        ),
    )

    assert result.verdict == "INSUFFICIENT_FOR_JUDGMENT"
