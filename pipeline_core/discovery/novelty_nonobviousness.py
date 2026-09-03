from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EvidenceState = Literal[
    "ESTABLISHED",
    "NOT_FOUND",
    "UNASSESSED",
]

BridgeKind = Literal[
    "NONE",
    "MAIN_EFFECTS_ONLY",
    "MEDIATION_CHAIN",
    "INTERACTION_COMPATIBLE",
]

StructuralNonObviousnessStatus = Literal[
    "DIRECTLY_KNOWN",
    "ROUTINE_COMPOSITION",
    "INTERACTION_LEAP",
    "MECHANISTIC_LEAP",
    "REGIME_OR_THRESHOLD_LEAP",
    "INSUFFICIENT_CLOSURE",
]


@dataclass(frozen=True)
class NonObviousnessEvidenceClosure:
    """Search-bounded evidence state around one residual claim.

    This object does not assert literature-wide absence.

    base_relation:
        The unconditioned scientific relation underlying the residue,
        e.g. spacing -> SERS.

    distinguishing_factor_effect:
        A lower-order effect involving the novelty-bearing factor,
        e.g. laser power -> SERS/local state.

    bridge_relation:
        A relation connecting the distinguishing factor to the
        variables/mechanism of the base relation,
        e.g. power -> gap or power -> relevant state.

    full_relation:
        The actual residual relation nucleus,
        e.g. power moderates spacing -> SERS.
    """

    base_relation: EvidenceState
    distinguishing_factor_effect: EvidenceState
    bridge_relation: EvidenceState
    full_relation: EvidenceState

    bridge_kind: BridgeKind = "NONE"
    scope_compatible: bool = True

    base_work_ids: tuple[str, ...] = ()
    factor_work_ids: tuple[str, ...] = ()
    bridge_work_ids: tuple[str, ...] = ()
    full_relation_work_ids: tuple[str, ...] = ()

    # Positive-only internal grounding provenance.
    #
    # These IDs may establish a positively grounded lower-order
    # relation, but they can never encode literature absence or
    # create a NOT_FOUND state.
    base_internal_statement_ids: tuple[str, ...] = ()
    factor_internal_statement_ids: tuple[str, ...] = ()
    bridge_internal_statement_ids: tuple[str, ...] = ()
    full_relation_internal_statement_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResidualClaimStructure:
    claim_kind: str

    introduces_new_mechanism: bool = False
    introduces_threshold: bool = False
    introduces_regime_change: bool = False
    introduces_reversal: bool = False
    introduces_mechanism_switch: bool = False


@dataclass(frozen=True)
class StructuralNonObviousnessAssessment:
    status: StructuralNonObviousnessStatus
    reason_codes: tuple[str, ...]
    interpretation: str


def assess_structural_nonobviousness(
    closure: NonObviousnessEvidenceClosure,
    structure: ResidualClaimStructure,
) -> StructuralNonObviousnessAssessment:
    """Classify the structure of the remaining inferential leap.

    This is not a scientific novelty score.

    In particular:
    - missing prior art is not automatically non-obviousness;
    - separate main effects do not establish an interaction;
    - a mediation chain does not automatically establish moderation.
    """

    if closure.full_relation == "ESTABLISHED":
        return StructuralNonObviousnessAssessment(
            status="DIRECTLY_KNOWN",
            reason_codes=(
                "full_residual_relation_established",
            ),
            interpretation=(
                "The residual relation itself is already positively "
                "represented in the reviewed prior art."
            ),
        )

    # A residual claim cannot enter structural non-obviousness
    # analysis while its FULL relation remains unassessed.
    #
    # This is deliberately asymmetric:
    #   ESTABLISHED -> directly known
    #   NOT_FOUND   -> search-bounded negative closure may proceed
    #   UNASSESSED  -> insufficient closure
    #
    # In particular, a threshold/regime structure must not become
    # REGIME_OR_THRESHOLD_LEAP merely because the base relation is
    # established while the full higher-order relation has not been
    # adequately searched.
    if closure.full_relation == "UNASSESSED":
        return StructuralNonObviousnessAssessment(
            status="INSUFFICIENT_CLOSURE",
            reason_codes=(
                "full_residual_relation_unassessed",
            ),
            interpretation=(
                "The full residual relation has not received sufficient "
                "prior-art closure. Structural non-obviousness analysis "
                "must stop until that relation is positively established "
                "or reaches a search-bounded NOT_FOUND state."
            ),
        )

    if not closure.scope_compatible:
        return StructuralNonObviousnessAssessment(
            status="INSUFFICIENT_CLOSURE",
            reason_codes=(
                "lower_order_evidence_scope_mismatch",
            ),
            interpretation=(
                "The available lower-order relations are not in a "
                "sufficiently compatible scientific scope to support "
                "a structural obviousness judgment."
            ),
        )

    # Stronger scientific structures remain distinct even when their
    # lower-order ingredients are known.
    if (
        structure.introduces_threshold
        or structure.introduces_regime_change
        or structure.introduces_reversal
        or structure.introduces_mechanism_switch
    ):
        if closure.base_relation != "ESTABLISHED":
            return StructuralNonObviousnessAssessment(
                status="INSUFFICIENT_CLOSURE",
                reason_codes=(
                    "base_relation_not_established",
                    "higher_order_structure_present",
                ),
                interpretation=(
                    "The claim contains a threshold, regime, reversal, "
                    "or mechanism-switch structure, but the lower-order "
                    "closure is not sufficiently established to assess "
                    "the inferential leap."
                ),
            )

        return StructuralNonObviousnessAssessment(
            status="REGIME_OR_THRESHOLD_LEAP",
            reason_codes=(
                "higher_order_regime_structure_not_inherited_from_components",
            ),
            interpretation=(
                "The residue introduces a threshold, regime change, "
                "reversal, or mechanism switch that is not supplied by "
                "ordinary component composition."
            ),
        )

    if structure.introduces_new_mechanism:
        if closure.base_relation != "ESTABLISHED":
            return StructuralNonObviousnessAssessment(
                status="INSUFFICIENT_CLOSURE",
                reason_codes=(
                    "base_relation_not_established",
                    "new_mechanism_claimed",
                ),
                interpretation=(
                    "A new mechanism is proposed, but the surrounding "
                    "lower-order evidence closure is not sufficiently "
                    "established for structural adjudication."
                ),
            )

        return StructuralNonObviousnessAssessment(
            status="MECHANISTIC_LEAP",
            reason_codes=(
                "new_mechanistic_bridge_not_inherited_from_components",
            ),
            interpretation=(
                "The residual claim requires a mechanistic bridge that "
                "is not already contained in the established component "
                "relations."
            ),
        )

    # A moderator interaction requires special treatment.
    if structure.claim_kind == "moderator_interaction":
        if closure.base_relation != "ESTABLISHED":
            return StructuralNonObviousnessAssessment(
                status="INSUFFICIENT_CLOSURE",
                reason_codes=(
                    "base_relation_not_established",
                ),
                interpretation=(
                    "The underlying unconditioned relation has not been "
                    "positively established, so the moderator residue "
                    "cannot yet be structurally adjudicated."
                ),
            )

        if (
            closure.distinguishing_factor_effect != "ESTABLISHED"
            or closure.bridge_relation != "ESTABLISHED"
        ):
            return StructuralNonObviousnessAssessment(
                status="INSUFFICIENT_CLOSURE",
                reason_codes=(
                    "distinguishing_factor_lower_order_closure_incomplete",
                ),
                interpretation=(
                    "The base relation is known, but the reviewed "
                    "literature does not yet establish enough lower-order "
                    "relations involving the moderator to distinguish a "
                    "routine extension from a genuine interaction leap."
                ),
            )

        if closure.bridge_kind in {
            "MAIN_EFFECTS_ONLY",
            "MEDIATION_CHAIN",
            "NONE",
        }:
            return StructuralNonObviousnessAssessment(
                status="INTERACTION_LEAP",
                reason_codes=(
                    "known_lower_order_relations_do_not_entail_interaction",
                ),
                interpretation=(
                    "Known main effects or a mediation chain do not by "
                    "themselves establish that the distinguishing factor "
                    "changes the base input-to-outcome relationship."
                ),
            )

        if closure.bridge_kind == "INTERACTION_COMPATIBLE":
            return StructuralNonObviousnessAssessment(
                status="ROUTINE_COMPOSITION",
                reason_codes=(
                    "interaction_compatible_bridge_already_established",
                ),
                interpretation=(
                    "The lower-order prior-art closure already contains "
                    "an interaction-compatible bridge, so the remaining "
                    "claim adds little beyond composition of known "
                    "relations."
                ),
            )

    # For non-interaction residuals, a fully established compatible
    # lower-order chain may be a routine composition.
    if (
        closure.base_relation == "ESTABLISHED"
        and closure.distinguishing_factor_effect == "ESTABLISHED"
        and closure.bridge_relation == "ESTABLISHED"
        and closure.bridge_kind
        in {
            "MEDIATION_CHAIN",
            "INTERACTION_COMPATIBLE",
        }
    ):
        return StructuralNonObviousnessAssessment(
            status="ROUTINE_COMPOSITION",
            reason_codes=(
                "compatible_lower_order_chain_established",
            ),
            interpretation=(
                "The residual proposition is structurally reconstructed "
                "from established, scope-compatible lower-order relations."
            ),
        )

    return StructuralNonObviousnessAssessment(
        status="INSUFFICIENT_CLOSURE",
        reason_codes=(
            "structural_evidence_closure_incomplete",
        ),
        interpretation=(
            "The reviewed evidence is insufficient to distinguish a "
            "routine composition from a scientifically non-obvious "
            "inferential leap."
        ),
    )
