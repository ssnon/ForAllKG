from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


InferentialDistance = Literal[
    "LOCAL_REPHRASE",
    "SINGLE_KNOWN_STEP",
    "MULTI_STEP_COMPOSITION",
    "NEW_RELATIONAL_FORM",
    "NEW_REGIME_STRUCTURE",
]

MechanisticNecessity = Literal[
    "NO_NEW_MECHANISM",
    "KNOWN_MECHANISM_REUSED",
    "NEW_BRIDGE_REQUIRED",
    "MECHANISM_SWITCH_REQUIRED",
]

RegimeSpecificity = Literal[
    "NONE",
    "CONDITIONED",
    "THRESHOLD",
    "REVERSAL",
    "HYSTERESIS",
    "MECHANISM_SWITCH",
]

Counterintuitiveness = Literal[
    "EXPECTED",
    "NONTRIVIAL",
    "COUNTER_TO_BASELINE",
]

TestableDistinctiveness = Literal[
    "GENERIC",
    "COMPARATIVE",
    "QUANTITATIVE",
    "DISCRIMINATING_SIGNATURE",
]

AdjudicationReadiness = Literal[
    "NOT_ELIGIBLE",
    "NEEDS_REFINEMENT",
    "READY_FOR_NONOBVIOUSNESS_REVIEW",
]


_ELIGIBLE_STRUCTURAL_STATUSES = {
    "INTERACTION_LEAP",
    "MECHANISTIC_LEAP",
    "REGIME_OR_THRESHOLD_LEAP",
}


@dataclass(frozen=True)
class NonObviousnessAdjudicationVector:
    """Scientific structure of one novelty residue.

    These dimensions are categorical and deliberately NOT summed into
    a scalar novelty score.

    inferential_distance:
        How far the claim moves beyond already established relations.

    mechanistic_necessity:
        Whether an additional causal/mechanistic bridge is required.

    regime_specificity:
        Whether the claim specifies a conditioned, threshold,
        reversal, hysteretic, or mechanism-switch regime.

    counterintuitiveness:
        Whether the predicted relation is merely expected,
        nontrivial, or counter to the natural baseline expectation.

    testable_distinctiveness:
        How specifically the claim distinguishes itself experimentally.
    """

    inferential_distance: InferentialDistance
    mechanistic_necessity: MechanisticNecessity
    regime_specificity: RegimeSpecificity
    counterintuitiveness: Counterintuitiveness
    testable_distinctiveness: TestableDistinctiveness

    required_bridge: str
    predicted_observation: str
    falsification_condition: str


@dataclass(frozen=True)
class NonObviousnessReviewGate:
    readiness: AdjudicationReadiness
    reason_codes: tuple[str, ...]
    interpretation: str


def assess_adjudication_readiness(
    *,
    structural_status: str,
    vector: NonObviousnessAdjudicationVector,
) -> NonObviousnessReviewGate:
    """Decide whether a candidate is specified enough for adjudication.

    This function does NOT decide that a hypothesis is scientifically
    non-obvious.

    It only distinguishes:
      1. claims not structurally eligible;
      2. under-specified claims that need refinement;
      3. sufficiently explicit candidates that can be reviewed.

    Missing direct prior art is never a positive signal by itself.
    """

    if structural_status not in _ELIGIBLE_STRUCTURAL_STATUSES:
        return NonObviousnessReviewGate(
            readiness="NOT_ELIGIBLE",
            reason_codes=(
                "structural_status_not_nonobviousness_candidate",
            ),
            interpretation=(
                "The structural analysis has not established an "
                "eligible inferential leap for non-obviousness review."
            ),
        )

    reasons: list[str] = []

    if vector.testable_distinctiveness == "GENERIC":
        reasons.append(
            "prediction_not_distinctively_testable"
        )

    if not vector.required_bridge.strip():
        reasons.append(
            "required_inferential_bridge_not_explicit"
        )

    if not vector.predicted_observation.strip():
        reasons.append(
            "predicted_observation_missing"
        )

    if not vector.falsification_condition.strip():
        reasons.append(
            "falsification_condition_missing"
        )

    # An interaction claim that remains only a generic conditioned
    # relation is not yet a scientifically informative novelty unit.
    if (
        structural_status == "INTERACTION_LEAP"
        and vector.inferential_distance
        in {
            "LOCAL_REPHRASE",
            "SINGLE_KNOWN_STEP",
        }
        and vector.mechanistic_necessity
        in {
            "NO_NEW_MECHANISM",
            "KNOWN_MECHANISM_REUSED",
        }
        and vector.regime_specificity
        in {
            "NONE",
            "CONDITIONED",
        }
    ):
        reasons.append(
            "interaction_candidate_lacks_distinctive_bridge_or_regime"
        )

    # A purported regime/threshold leap should actually contain
    # a higher-order regime structure.
    if (
        structural_status
        == "REGIME_OR_THRESHOLD_LEAP"
        and vector.regime_specificity
        not in {
            "THRESHOLD",
            "REVERSAL",
            "HYSTERESIS",
            "MECHANISM_SWITCH",
        }
    ):
        reasons.append(
            "regime_leap_without_specific_regime_structure"
        )

    # A mechanistic leap must require more than reuse of an already
    # established mechanism.
    if (
        structural_status == "MECHANISTIC_LEAP"
        and vector.mechanistic_necessity
        not in {
            "NEW_BRIDGE_REQUIRED",
            "MECHANISM_SWITCH_REQUIRED",
        }
    ):
        reasons.append(
            "mechanistic_leap_without_new_mechanistic_requirement"
        )

    if reasons:
        return NonObviousnessReviewGate(
            readiness="NEEDS_REFINEMENT",
            reason_codes=tuple(reasons),
            interpretation=(
                "The claim is structurally interesting but not yet "
                "specified strongly enough for a credible scientific "
                "non-obviousness adjudication."
            ),
        )

    return NonObviousnessReviewGate(
        readiness="READY_FOR_NONOBVIOUSNESS_REVIEW",
        reason_codes=(
            "explicit_bridge_prediction_and_falsifier_present",
        ),
        interpretation=(
            "The candidate contains an explicit inferential bridge, "
            "a distinctive prediction, and a falsification condition. "
            "It is ready for scientific non-obviousness review, but "
            "has not been judged non-obvious."
        ),
    )



# N9-C2b evidence-constrained adjudication

FinalAdjudicationVerdict = Literal[
    "ROUTINE_FROM_PRIOR_ART",
    "INSUFFICIENT_FOR_JUDGMENT",
    "POTENTIALLY_NON_OBVIOUS",
]


@dataclass(frozen=True)
class EstablishedPriorArtRelation:
    """One positively reviewed relation available to the adjudicator."""

    relation_statement: str
    relationship_status: str
    work_ids: tuple[str, ...]
    scope_note: str = ""


@dataclass(frozen=True)
class NonObviousnessEvidencePacket:
    """Evidence packet for one review-ready novelty residue.

    Only positively reviewed prior-art relations belong in
    established_relations. Missing search matches are not positive evidence.

    The packet keeps the residual claim separate from the literature-backed
    relations so the adjudicator must explicitly identify any additional
    scientific step needed to reach the claim.
    """

    claim_id: str
    claim_text: str
    structural_status: str

    vector: NonObviousnessAdjudicationVector

    established_relations: tuple[
        EstablishedPriorArtRelation,
        ...
    ]

    direct_full_claim_prior_art: bool
    evidence_closure_sufficient: bool


@dataclass(frozen=True)
class NonObviousnessAdjudicationDraft:
    """Structured reviewer output.

    This is intentionally not free-form novelty scoring.
    """

    proposed_verdict: FinalAdjudicationVerdict

    direct_reconstruction_from_known_relations: bool

    additional_scientific_assumptions: tuple[str, ...]

    prediction_distinguishes_from_routine_baseline: bool
    falsifier_is_specific: bool

    concise_basis: str


@dataclass(frozen=True)
class CompiledNonObviousnessAdjudication:
    verdict: FinalAdjudicationVerdict
    reason_codes: tuple[str, ...]
    required_additional_assumptions: tuple[str, ...]
    interpretation: str


def compile_nonobviousness_adjudication(
    *,
    readiness: NonObviousnessReviewGate,
    packet: NonObviousnessEvidencePacket,
    draft: NonObviousnessAdjudicationDraft,
) -> CompiledNonObviousnessAdjudication:
    """Compile a conservative final non-obviousness disposition.

    The compiler is deliberately asymmetric:

    POTENTIALLY_NON_OBVIOUS requires multiple positive structural
    conditions.

    Failure to meet those conditions does NOT default to routine;
    it defaults to INSUFFICIENT_FOR_JUDGMENT unless the known prior
    art positively reconstructs the claim.
    """

    # Positive evidence that the residual claim is already known or
    # structurally routine takes precedence over non-obviousness readiness.
    #
    # Readiness answers whether a candidate may enter NON-OBVIOUSNESS
    # review. It must not suppress a positive ROUTINE determination that
    # has already been established by prior-art closure.
    if (
        packet.direct_full_claim_prior_art
        or packet.structural_status == "DIRECTLY_KNOWN"
    ):
        return CompiledNonObviousnessAdjudication(
            verdict="ROUTINE_FROM_PRIOR_ART",
            reason_codes=(
                "full_claim_prior_art_already_established",
            ),
            required_additional_assumptions=(),
            interpretation=(
                "The full residual claim is already positively represented "
                "in the reviewed prior art."
            ),
        )

    if packet.structural_status == "ROUTINE_COMPOSITION":
        return CompiledNonObviousnessAdjudication(
            verdict="ROUTINE_FROM_PRIOR_ART",
            reason_codes=(
                "structural_routine_composition_established",
            ),
            required_additional_assumptions=(),
            interpretation=(
                "The reviewed, scope-compatible prior-art closure "
                "structurally reconstructs the residual claim as a routine "
                "composition of established relations."
            ),
        )

    if readiness.readiness != "READY_FOR_NONOBVIOUSNESS_REVIEW":
        return CompiledNonObviousnessAdjudication(
            verdict="INSUFFICIENT_FOR_JUDGMENT",
            reason_codes=(
                "candidate_not_ready_for_adjudication",
            ),
            required_additional_assumptions=(),
            interpretation=(
                "The candidate has not passed the structural/specification "
                "gate required for scientific non-obviousness adjudication."
            ),
        )

    if not packet.evidence_closure_sufficient:
        return CompiledNonObviousnessAdjudication(
            verdict="INSUFFICIENT_FOR_JUDGMENT",
            reason_codes=(
                "evidence_closure_insufficient",
            ),
            required_additional_assumptions=(),
            interpretation=(
                "The lower-order prior-art closure is insufficient for a "
                "credible routine-versus-non-obviousness judgment."
            ),
        )

    if draft.direct_reconstruction_from_known_relations:
        return CompiledNonObviousnessAdjudication(
            verdict="ROUTINE_FROM_PRIOR_ART",
            reason_codes=(
                "claim_reconstructed_from_established_relations",
            ),
            required_additional_assumptions=(),
            interpretation=(
                "The reviewer reports that the residual claim follows "
                "directly from the established, scope-compatible prior-art "
                "relations without an additional scientific bridge."
            ),
        )

    potential_requirements = {
        "additional_assumption": bool(
            tuple(
                row.strip()
                for row in draft.additional_scientific_assumptions
                if row.strip()
            )
        ),
        "distinctive_prediction": (
            draft.prediction_distinguishes_from_routine_baseline
        ),
        "specific_falsifier": draft.falsifier_is_specific,
        "basis_present": bool(draft.concise_basis.strip()),
    }

    if (
        draft.proposed_verdict == "POTENTIALLY_NON_OBVIOUS"
        and all(potential_requirements.values())
    ):
        assumptions = tuple(
            row.strip()
            for row in draft.additional_scientific_assumptions
            if row.strip()
        )

        return CompiledNonObviousnessAdjudication(
            verdict="POTENTIALLY_NON_OBVIOUS",
            reason_codes=(
                "not_directly_reconstructable_from_known_relations",
                "additional_scientific_bridge_required",
                "prediction_distinguishes_from_routine_baseline",
                "specific_falsification_condition_present",
            ),
            required_additional_assumptions=assumptions,
            interpretation=(
                "Under the supplied, search-bounded prior-art closure, the "
                "claim requires an explicit additional scientific bridge and "
                "makes a prediction that distinguishes it from routine "
                "composition. This is a candidate non-obviousness judgment, "
                "not proof of literature-wide novelty or scientific truth."
            ),
        )

    missing = tuple(
        name
        for name, present in potential_requirements.items()
        if not present
    )

    return CompiledNonObviousnessAdjudication(
        verdict="INSUFFICIENT_FOR_JUDGMENT",
        reason_codes=(
            "nonobviousness_requirements_not_satisfied",
            *tuple(
                f"missing_{name}"
                for name in missing
            ),
        ),
        required_additional_assumptions=tuple(
            row.strip()
            for row in draft.additional_scientific_assumptions
            if row.strip()
        ),
        interpretation=(
            "The reviewed evidence does not support direct reconstruction "
            "of the claim, but the reviewer has also not supplied enough "
            "specific scientific structure to justify a potentially "
            "non-obvious classification."
        ),
    )
