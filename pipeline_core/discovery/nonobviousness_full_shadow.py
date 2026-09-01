from __future__ import annotations

from dataclasses import dataclass

from pipeline_core.discovery.novelty_adjudication import (
    CompiledNonObviousnessAdjudication,
    NonObviousnessAdjudicationDraft,
    NonObviousnessAdjudicationVector,
    NonObviousnessEvidencePacket,
    NonObviousnessReviewGate,
    compile_nonobviousness_adjudication,
)
from pipeline_core.discovery.novelty_nonobviousness import (
    ResidualClaimStructure,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
)


@dataclass(frozen=True)
class ConservativeNonObviousnessInputs:
    """Fail-closed structural inputs when no explicit mapper exists.

    We preserve claim_kind and exact scientific specification text.
    Stronger structural categories are NOT inferred from free text.
    Scope compatibility and bridge-kind compatibility remain unassessed.
    """

    structure: ResidualClaimStructure
    vector: NonObviousnessAdjudicationVector
    bridge_kind: str
    scope_compatible: bool
    reason_codes: tuple[str, ...]


def derive_conservative_nonobviousness_inputs(
    claim: NoveltyResidueClaim,
) -> ConservativeNonObviousnessInputs:
    structure = ResidualClaimStructure(
        claim_kind=claim.claim_kind,
        introduces_new_mechanism=False,
        introduces_threshold=False,
        introduces_regime_change=False,
        introduces_reversal=False,
        introduces_mechanism_switch=False,
    )

    vector = NonObviousnessAdjudicationVector(
        inferential_distance="LOCAL_REPHRASE",
        mechanistic_necessity="NO_NEW_MECHANISM",
        regime_specificity="NONE",
        counterintuitiveness="EXPECTED",
        testable_distinctiveness="GENERIC",
        required_bridge=claim.required_bridge,
        predicted_observation=claim.predicted_observation,
        falsification_condition=claim.falsification_condition,
    )

    return ConservativeNonObviousnessInputs(
        structure=structure,
        vector=vector,
        bridge_kind="NONE",
        scope_compatible=False,
        reason_codes=(
            "structure_derived_from_claim_kind_only",
            "higher_order_structure_not_inferred_from_text",
            "bridge_kind_unassessed_default_none",
            "scope_compatibility_unassessed_fail_closed",
            "adjudication_vector_categories_unassessed_conservative_defaults",
        ),
    )


def neutral_adjudication_draft(
) -> NonObviousnessAdjudicationDraft:
    """Placeholder used only when compiler outcome is already forced.

    It must never be used to adjudicate a READY candidate.
    """

    return NonObviousnessAdjudicationDraft(
        proposed_verdict="INSUFFICIENT_FOR_JUDGMENT",
        direct_reconstruction_from_known_relations=False,
        additional_scientific_assumptions=(),
        prediction_distinguishes_from_routine_baseline=False,
        falsifier_is_specific=False,
        concise_basis="",
    )


def compile_forced_adjudication_if_determined(
    *,
    readiness: NonObviousnessReviewGate,
    packet: NonObviousnessEvidencePacket,
) -> tuple[
    str,
    CompiledNonObviousnessAdjudication | None,
]:
    """Compile only verdicts determined before independent adjudication.

    NOT_ELIGIBLE / NEEDS_REFINEMENT outcomes are deterministic.
    DIRECTLY_KNOWN and ROUTINE_COMPOSITION are also deterministic due
    the positive-routine short circuit.

    READY candidates require an independent adjudicator and are not
    assigned a synthetic final verdict here.
    """

    if (
        readiness.readiness
        == "READY_FOR_NONOBVIOUSNESS_REVIEW"
    ):
        return (
            "PENDING_INDEPENDENT_ADJUDICATOR",
            None,
        )

    result = compile_nonobviousness_adjudication(
        readiness=readiness,
        packet=packet,
        draft=neutral_adjudication_draft(),
    )

    return (
        "COMPILED_DETERMINISTIC_SHORT_CIRCUIT",
        result,
    )
