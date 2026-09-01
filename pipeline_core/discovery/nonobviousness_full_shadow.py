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
    """Map validated atomic structure into N9 structural inputs.

    The historical function name is retained for API compatibility.

    Scientific structure/vector categories now come from the
    provenance-validated atomic NoveltyClaim contract. They are never
    inferred from free text here.

    Evidence-side bridge kind and scope compatibility deliberately
    remain conservative until N10-B compiles them from reviewed
    prior-art closure.
    """

    declared = claim.scientific_structure

    structure = ResidualClaimStructure(
        claim_kind=claim.claim_kind,
        introduces_new_mechanism=(
            declared.introduces_new_mechanism
        ),
        introduces_threshold=(
            declared.introduces_threshold
        ),
        introduces_regime_change=(
            declared.introduces_regime_change
        ),
        introduces_reversal=(
            declared.introduces_reversal
        ),
        introduces_mechanism_switch=(
            declared.introduces_mechanism_switch
        ),
    )

    vector = NonObviousnessAdjudicationVector(
        inferential_distance=(
            declared.inferential_distance
        ),
        mechanistic_necessity=(
            declared.mechanistic_necessity
        ),
        regime_specificity=(
            declared.regime_specificity
        ),
        counterintuitiveness=(
            declared.counterintuitiveness
        ),
        testable_distinctiveness=(
            declared.testable_distinctiveness
        ),
        required_bridge=claim.required_bridge,
        predicted_observation=(
            claim.predicted_observation
        ),
        falsification_condition=(
            claim.falsification_condition
        ),
    )

    has_explicit_structure = bool(
        declared.basis
        or declared.introduces_new_mechanism
        or declared.introduces_threshold
        or declared.introduces_regime_change
        or declared.introduces_reversal
        or declared.introduces_mechanism_switch
        or declared.inferential_distance
        != "LOCAL_REPHRASE"
        or declared.mechanistic_necessity
        != "NO_NEW_MECHANISM"
        or declared.regime_specificity
        != "NONE"
        or declared.counterintuitiveness
        != "EXPECTED"
        or declared.testable_distinctiveness
        != "GENERIC"
    )

    if has_explicit_structure:
        structure_reasons = (
            "atomic_scientific_structure_provenance_validated",
            *claim.scientific_structure_reason_codes,
        )
    else:
        # Preserve the old observable contract for ordinary/default
        # claims and existing regression fixtures.
        structure_reasons = (
            "structure_derived_from_claim_kind_only",
            "higher_order_structure_not_inferred_from_text",
            *claim.scientific_structure_reason_codes,
        )

    return ConservativeNonObviousnessInputs(
        structure=structure,
        vector=vector,

        # Evidence-side placeholders, intentionally unchanged here.
        bridge_kind="NONE",
        scope_compatible=False,

        reason_codes=(
            *structure_reasons,
            "bridge_kind_unassessed_default_none",
            "scope_compatibility_unassessed_fail_closed",
            (
                "adjudication_vector_from_validated_atomic_structure"
                if has_explicit_structure
                else
                "adjudication_vector_categories_unassessed_conservative_defaults"
            ),
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
