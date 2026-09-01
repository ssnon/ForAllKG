from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
    NoveltyStructureBasis,
)
from pipeline_core.discovery.nonobviousness_full_shadow import (
    derive_conservative_nonobviousness_inputs,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
)


def _claim(
    *,
    structure=None,
    structure_reasons=(),
):
    return NoveltyResidueClaim(
        hypothesis_id="hypothesis:h",
        claim_id="claim:c",
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
            "Below and above laser power Pc, the spacing-to-SERS "
            "response occupies two distinguishable regimes."
        ),
        falsification_condition=(
            "The spacing-to-SERS response varies smoothly with "
            "laser power and shows no reproducible regime boundary."
        ),
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
        scientific_structure=(
            structure
            or NoveltyClaimScientificStructure()
        ),
        scientific_structure_reason_codes=tuple(
            structure_reasons
        ),
    )


def test_validated_threshold_structure_maps_into_n9_inputs():
    structure = NoveltyClaimScientificStructure(
        introduces_threshold=True,
        introduces_regime_change=True,
        inferential_distance="NEW_REGIME_STRUCTURE",
        mechanistic_necessity="NEW_BRIDGE_REQUIRED",
        regime_specificity="THRESHOLD",
        counterintuitiveness="NONTRIVIAL",
        testable_distinctiveness="QUANTITATIVE",
        basis=[
            NoveltyStructureBasis(
                feature="threshold",
                source_text=(
                    "A critical laser power Pc separates two "
                    "distinct spacing-to-SERS regimes."
                ),
            ),
        ],
    )

    result = derive_conservative_nonobviousness_inputs(
        _claim(structure=structure)
    )

    assert result.structure.introduces_threshold is True
    assert result.structure.introduces_regime_change is True

    assert (
        result.vector.inferential_distance
        == "NEW_REGIME_STRUCTURE"
    )
    assert (
        result.vector.mechanistic_necessity
        == "NEW_BRIDGE_REQUIRED"
    )
    assert (
        result.vector.regime_specificity
        == "THRESHOLD"
    )
    assert (
        result.vector.counterintuitiveness
        == "NONTRIVIAL"
    )
    assert (
        result.vector.testable_distinctiveness
        == "QUANTITATIVE"
    )

    assert (
        "atomic_scientific_structure_provenance_validated"
        in result.reason_codes
    )
    assert (
        "adjudication_vector_from_validated_atomic_structure"
        in result.reason_codes
    )

    # N10-B has not run yet.
    assert result.bridge_kind == "NONE"
    assert result.scope_compatible is False


def test_default_atomic_structure_preserves_old_fail_closed_behavior():
    result = derive_conservative_nonobviousness_inputs(
        _claim()
    )

    assert result.structure.introduces_threshold is False
    assert result.structure.introduces_regime_change is False
    assert (
        result.vector.inferential_distance
        == "LOCAL_REPHRASE"
    )
    assert result.vector.regime_specificity == "NONE"
    assert (
        result.vector.testable_distinctiveness
        == "GENERIC"
    )

    assert (
        "higher_order_structure_not_inferred_from_text"
        in result.reason_codes
    )
    assert (
        "adjudication_vector_categories_unassessed_conservative_defaults"
        in result.reason_codes
    )


def test_structure_validation_reason_codes_propagate_to_n9():
    result = derive_conservative_nonobviousness_inputs(
        _claim(
            structure_reasons=(
                "unsupported_structure_flag:threshold",
            )
        )
    )

    # No supported strong category remains.
    assert result.structure.introduces_threshold is False

    assert (
        "unsupported_structure_flag:threshold"
        in result.reason_codes
    )
