from pipeline_core.discovery.novelty_adjudication import (
    EstablishedPriorArtRelation,
    NonObviousnessAdjudicationVector,
    NonObviousnessEvidencePacket,
)
from pipeline_core.discovery.novelty_adjudication_llm import (
    _NONOBVIOUSNESS_SYSTEM,
    build_nonobviousness_user_prompt,
)


def packet():
    return NonObviousnessEvidencePacket(
        claim_id="claim:test",
        claim_text=(
            "Laser power moderates the spacing-to-SERS relation."
        ),
        structural_status="INTERACTION_LEAP",
        vector=NonObviousnessAdjudicationVector(
            inferential_distance="NEW_RELATIONAL_FORM",
            mechanistic_necessity="NEW_BRIDGE_REQUIRED",
            regime_specificity="CONDITIONED",
            counterintuitiveness="NONTRIVIAL",
            testable_distinctiveness="COMPARATIVE",
            required_bridge=(
                "A power-sensitive interaction must modify "
                "the spacing response."
            ),
            predicted_observation=(
                "Spacing-to-SERS curves differ across power."
            ),
            falsification_condition=(
                "The spacing-to-SERS relation is invariant "
                "across power."
            ),
        ),
        established_relations=(
            EstablishedPriorArtRelation(
                relation_statement=(
                    "Interparticle spacing affects SERS."
                ),
                relationship_status="DIRECT_PRIOR_ART",
                work_ids=("w1",),
            ),
            EstablishedPriorArtRelation(
                relation_statement=(
                    "Laser power affects measured SERS intensity."
                ),
                relationship_status="DIRECT_PRIOR_ART",
                work_ids=("w2",),
            ),
        ),
        direct_full_claim_prior_art=False,
        evidence_closure_sufficient=True,
    )


def test_prompt_forbids_missing_match_novelty():
    assert (
        "Do not treat failure to find an exact paper as evidence"
        in _NONOBVIOUSNESS_SYSTEM
    )


def test_prompt_forbids_invented_mechanism():
    assert (
        "You MUST NOT invent a new mechanism"
        in _NONOBVIOUSNESS_SYSTEM
    )


def test_prompt_states_main_effect_non_entailment():
    assert (
        "Separate main effects do not establish an interaction."
        in _NONOBVIOUSNESS_SYSTEM
    )


def test_prompt_requires_insufficient_when_bridge_unspecified():
    assert (
        "return INSUFFICIENT_FOR_JUDGMENT"
        in _NONOBVIOUSNESS_SYSTEM
    )


def test_user_prompt_keeps_claim_and_prior_art_separate():
    text = build_nonobviousness_user_prompt(
        packet()
    )

    assert "RESIDUAL CLAIM" in text
    assert "ESTABLISHED PRIOR-ART RELATIONS" in text

    assert (
        "Interparticle spacing affects SERS."
        in text
    )

    assert (
        "Laser power affects measured SERS intensity."
        in text
    )
