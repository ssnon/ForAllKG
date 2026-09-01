from pipeline_core.discovery.nonobviousness_full_shadow import (
    compile_forced_adjudication_if_determined,
    derive_conservative_nonobviousness_inputs,
)
from pipeline_core.discovery.novelty_adjudication import (
    NonObviousnessEvidencePacket,
    NonObviousnessReviewGate,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
)


def threshold_like_claim():
    return NoveltyResidueClaim(
        hypothesis_id="hypothesis:threshold",
        claim_id="claim:threshold",
        claim_text=(
            "A critical laser power Pc separates two distinct "
            "spacing-to-SERS regimes."
        ),
        claim_kind="distinctive_prediction",
        prior_art_status="NO_DIRECT_MATCH_FOUND",
        disposition="RESIDUAL",
        is_residue=True,
        distinguishing_terms=(
            "critical laser power",
            "two regimes",
        ),
        prior_art_identity_terms=("laser power",),
        relation_nucleus_terms=(
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ),
        required_bridge=(
            "Laser power drives a transition at Pc that changes "
            "how spacing maps to measured SERS enhancement."
        ),
        predicted_observation=(
            "Below and above Pc, the spacing-to-SERS response "
            "occupies two distinguishable regimes."
        ),
        falsification_condition=(
            "The spacing-to-SERS response varies smoothly with "
            "power and shows no reproducible regime boundary."
        ),
        direct_or_partial_work_ids=(),
        lower_order_work_ids=("work:spacing-sers",),
        component_work_ids=("work:power",),
    )


def test_free_text_threshold_is_not_inferred():
    claim = threshold_like_claim()

    result = (
        derive_conservative_nonobviousness_inputs(
            claim
        )
    )

    assert result.structure.claim_kind == (
        "distinctive_prediction"
    )

    assert not result.structure.introduces_threshold
    assert not result.structure.introduces_regime_change
    assert not result.structure.introduces_new_mechanism

    assert result.scope_compatible is False
    assert result.bridge_kind == "NONE"

    assert (
        result.vector.required_bridge
        == claim.required_bridge
    )

    assert (
        result.vector.predicted_observation
        == claim.predicted_observation
    )

    assert (
        result.vector.falsification_condition
        == claim.falsification_condition
    )

    assert (
        result.vector.testable_distinctiveness
        == "GENERIC"
    )


def test_not_eligible_can_be_compiled_without_reviewer():
    claim = threshold_like_claim()

    inputs = (
        derive_conservative_nonobviousness_inputs(
            claim
        )
    )

    gate = NonObviousnessReviewGate(
        readiness="NOT_ELIGIBLE",
        reason_codes=(
            "structural_status_not_nonobviousness_candidate",
        ),
        interpretation="not eligible",
    )

    packet = NonObviousnessEvidencePacket(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        structural_status="INSUFFICIENT_CLOSURE",
        vector=inputs.vector,
        established_relations=(),
        direct_full_claim_prior_art=False,
        evidence_closure_sufficient=False,
    )

    status, final = (
        compile_forced_adjudication_if_determined(
            readiness=gate,
            packet=packet,
        )
    )

    assert status == (
        "COMPILED_DETERMINISTIC_SHORT_CIRCUIT"
    )

    assert final is not None
    assert final.verdict == (
        "INSUFFICIENT_FOR_JUDGMENT"
    )


def test_ready_candidate_is_not_given_placeholder_verdict():
    claim = threshold_like_claim()

    inputs = (
        derive_conservative_nonobviousness_inputs(
            claim
        )
    )

    gate = NonObviousnessReviewGate(
        readiness="READY_FOR_NONOBVIOUSNESS_REVIEW",
        reason_codes=(
            "explicit_bridge_prediction_and_falsifier_present",
        ),
        interpretation="ready",
    )

    packet = NonObviousnessEvidencePacket(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        structural_status="INTERACTION_LEAP",
        vector=inputs.vector,
        established_relations=(),
        direct_full_claim_prior_art=False,
        evidence_closure_sufficient=True,
    )

    status, final = (
        compile_forced_adjudication_if_determined(
            readiness=gate,
            packet=packet,
        )
    )

    assert status == (
        "PENDING_INDEPENDENT_ADJUDICATOR"
    )
    assert final is None
