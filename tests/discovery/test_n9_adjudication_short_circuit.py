from pipeline_core.discovery.novelty_adjudication import (
    NonObviousnessAdjudicationDraft,
    NonObviousnessAdjudicationVector,
    NonObviousnessEvidencePacket,
    assess_adjudication_readiness,
    compile_nonobviousness_adjudication,
)


def vector():
    return NonObviousnessAdjudicationVector(
        inferential_distance="LOCAL_REPHRASE",
        mechanistic_necessity="NO_NEW_MECHANISM",
        regime_specificity="NONE",
        counterintuitiveness="EXPECTED",
        testable_distinctiveness="GENERIC",
        required_bridge="",
        predicted_observation="",
        falsification_condition="",
    )


def draft():
    return NonObviousnessAdjudicationDraft(
        proposed_verdict="INSUFFICIENT_FOR_JUDGMENT",
        direct_reconstruction_from_known_relations=False,
        additional_scientific_assumptions=(),
        prediction_distinguishes_from_routine_baseline=False,
        falsifier_is_specific=False,
        concise_basis="",
    )


def test_directly_known_short_circuits_not_eligible_readiness():
    v = vector()

    readiness = assess_adjudication_readiness(
        structural_status="DIRECTLY_KNOWN",
        vector=v,
    )

    assert readiness.readiness == "NOT_ELIGIBLE"

    packet = NonObviousnessEvidencePacket(
        claim_id="claim:direct",
        claim_text="Known full residual relation.",
        structural_status="DIRECTLY_KNOWN",
        vector=v,
        established_relations=(),
        direct_full_claim_prior_art=True,
        evidence_closure_sufficient=True,
    )

    result = compile_nonobviousness_adjudication(
        readiness=readiness,
        packet=packet,
        draft=draft(),
    )

    assert result.verdict == "ROUTINE_FROM_PRIOR_ART"

    assert result.reason_codes == (
        "full_claim_prior_art_already_established",
    )


def test_routine_composition_short_circuits_not_eligible_readiness():
    v = vector()

    readiness = assess_adjudication_readiness(
        structural_status="ROUTINE_COMPOSITION",
        vector=v,
    )

    assert readiness.readiness == "NOT_ELIGIBLE"

    packet = NonObviousnessEvidencePacket(
        claim_id="claim:routine",
        claim_text="Routine composition.",
        structural_status="ROUTINE_COMPOSITION",
        vector=v,
        established_relations=(),
        direct_full_claim_prior_art=False,
        evidence_closure_sufficient=True,
    )

    result = compile_nonobviousness_adjudication(
        readiness=readiness,
        packet=packet,
        draft=draft(),
    )

    assert result.verdict == "ROUTINE_FROM_PRIOR_ART"

    assert result.reason_codes == (
        "structural_routine_composition_established",
    )


def test_insufficient_closure_remains_fail_closed():
    v = vector()

    readiness = assess_adjudication_readiness(
        structural_status="INSUFFICIENT_CLOSURE",
        vector=v,
    )

    assert readiness.readiness == "NOT_ELIGIBLE"

    packet = NonObviousnessEvidencePacket(
        claim_id="claim:unassessed",
        claim_text="Insufficiently closed residual relation.",
        structural_status="INSUFFICIENT_CLOSURE",
        vector=v,
        established_relations=(),
        direct_full_claim_prior_art=False,
        evidence_closure_sufficient=False,
    )

    result = compile_nonobviousness_adjudication(
        readiness=readiness,
        packet=packet,
        draft=draft(),
    )

    assert result.verdict == "INSUFFICIENT_FOR_JUDGMENT"

    assert result.reason_codes == (
        "candidate_not_ready_for_adjudication",
    )
