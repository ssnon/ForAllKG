from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
    assess_residual_specification,
)
from pipeline_core.discovery.novelty_nonobviousness import (
    NonObviousnessEvidenceClosure,
    ResidualClaimStructure,
    assess_structural_nonobviousness,
)
from pipeline_core.discovery.novelty_adjudication import (
    NonObviousnessAdjudicationVector,
    assess_adjudication_readiness,
)


def test_generic_power_residue_is_blocked_before_closure():
    claim = NoveltyResidueClaim(
        hypothesis_id="hypothesis:h2",
        claim_id="claim:power",
        claim_text=(
            "Laser power moderates the dependence of SERS "
            "enhancement on interparticle spacing."
        ),
        claim_kind="moderator_interaction",
        prior_art_status="COMPONENTS_ONLY",
        disposition="RESIDUAL",
        is_residue=True,
        distinguishing_terms=("laser power",),
        prior_art_identity_terms=("laser power",),
        relation_nucleus_terms=(
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ),
        required_bridge="",
        predicted_observation=(
            "The spacing-SERS relationship differs across "
            "laser powers."
        ),
        falsification_condition=(
            "The spacing-SERS relationship is indistinguishable "
            "across laser powers."
        ),
        direct_or_partial_work_ids=(),
        lower_order_work_ids=("work:spacing-sers",),
        component_work_ids=(),
    )

    result = assess_residual_specification(claim)

    assert result.status == "NEEDS_REFINEMENT"
    assert result.missing_fields == ("required_bridge",)


def test_explicit_threshold_control_survives_to_review_readiness():
    claim = NoveltyResidueClaim(
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
        prior_art_identity_terms=(
            "laser power",
        ),
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

    specification = assess_residual_specification(claim)

    assert specification.status == "READY_FOR_CLOSURE"

    structural = assess_structural_nonobviousness(
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
            introduces_regime_change=True,
        ),
    )

    assert structural.status == "REGIME_OR_THRESHOLD_LEAP"

    readiness = assess_adjudication_readiness(
        structural_status=structural.status,
        vector=NonObviousnessAdjudicationVector(
            inferential_distance="NEW_REGIME_STRUCTURE",
            mechanistic_necessity="NEW_BRIDGE_REQUIRED",
            regime_specificity="THRESHOLD",
            counterintuitiveness="NONTRIVIAL",
            testable_distinctiveness="QUANTITATIVE",
            required_bridge=claim.required_bridge,
            predicted_observation=claim.predicted_observation,
            falsification_condition=claim.falsification_condition,
        ),
    )

    assert (
        readiness.readiness
        == "READY_FOR_NONOBVIOUSNESS_REVIEW"
    )
