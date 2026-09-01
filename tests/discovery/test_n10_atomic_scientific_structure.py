from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
    NoveltyStructureBasis,
)
from pipeline_core.discovery.novelty_structure_validation import (
    compile_claim_scientific_structure,
)


def test_explicit_threshold_structure_with_source_basis_survives():
    statement = (
        "A critical laser power Pc separates two distinct "
        "spacing-to-SERS regimes."
    )
    bridge = (
        "Laser power drives a transition at Pc that changes how "
        "spacing maps to measured SERS enhancement."
    )
    prediction = (
        "Below and above laser power Pc, the spacing-to-SERS "
        "response occupies two distinguishable regimes."
    )

    draft = NoveltyClaimScientificStructure(
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
                source_text=statement,
            ),
            NoveltyStructureBasis(
                feature="regime_change",
                source_text=statement,
            ),
            NoveltyStructureBasis(
                feature="inferential_distance",
                source_text=statement,
            ),
            NoveltyStructureBasis(
                feature="mechanistic_necessity",
                source_text=bridge,
            ),
            NoveltyStructureBasis(
                feature="regime_specificity",
                source_text=statement,
            ),
            NoveltyStructureBasis(
                feature="counterintuitiveness",
                source_text=statement,
            ),
            NoveltyStructureBasis(
                feature="testable_distinctiveness",
                source_text=prediction,
            ),
        ],
    )

    result, reasons = compile_claim_scientific_structure(
        draft,
        identity_terms=["laser power"],
        source_texts=[
            statement,
            bridge,
            prediction,
        ],
    )

    assert result.introduces_threshold is True
    assert result.introduces_regime_change is True
    assert result.inferential_distance == "NEW_REGIME_STRUCTURE"
    assert result.mechanistic_necessity == "NEW_BRIDGE_REQUIRED"
    assert result.regime_specificity == "THRESHOLD"
    assert result.counterintuitiveness == "NONTRIVIAL"
    assert result.testable_distinctiveness == "QUANTITATIVE"
    assert reasons == ()


def test_unsupported_generic_power_threshold_fails_closed():
    source = (
        "Laser power moderates the dependence of SERS enhancement "
        "on interparticle spacing."
    )

    draft = NoveltyClaimScientificStructure(
        introduces_threshold=True,
        inferential_distance="NEW_REGIME_STRUCTURE",
        regime_specificity="THRESHOLD",
        basis=[
            NoveltyStructureBasis(
                feature="threshold",
                source_text=(
                    "A critical laser power threshold separates "
                    "two response regimes."
                ),
            ),
            NoveltyStructureBasis(
                feature="inferential_distance",
                source_text=(
                    "A critical laser power threshold separates "
                    "two response regimes."
                ),
            ),
            NoveltyStructureBasis(
                feature="regime_specificity",
                source_text=(
                    "A critical laser power threshold separates "
                    "two response regimes."
                ),
            ),
        ],
    )

    result, reasons = compile_claim_scientific_structure(
        draft,
        identity_terms=["laser power"],
        source_texts=[source],
    )

    assert result.introduces_threshold is False
    assert result.regime_specificity == "NONE"
    assert result.inferential_distance == "LOCAL_REPHRASE"
    assert any(
        x.startswith("structure_basis_not_extractive:")
        for x in reasons
    )


def test_sibling_branch_structure_basis_is_rejected():
    wavelength_source = (
        "Excitation wavelength introduces a threshold in the "
        "spacing-to-SERS response."
    )

    draft = NoveltyClaimScientificStructure(
        introduces_threshold=True,
        regime_specificity="THRESHOLD",
        basis=[
            NoveltyStructureBasis(
                feature="threshold",
                source_text=wavelength_source,
            ),
            NoveltyStructureBasis(
                feature="regime_specificity",
                source_text=wavelength_source,
            ),
        ],
    )

    result, reasons = compile_claim_scientific_structure(
        draft,
        identity_terms=["laser power"],
        source_texts=[wavelength_source],
    )

    assert result.introduces_threshold is False
    assert result.regime_specificity == "NONE"
    assert (
        "structure_basis_branch_identity_missing:threshold"
        in reasons
    )


def test_empty_structure_remains_conservative_without_penalty():
    result, reasons = compile_claim_scientific_structure(
        NoveltyClaimScientificStructure(),
        identity_terms=["laser power"],
        source_texts=[
            "Laser power moderates a spacing-dependent response."
        ],
    )

    assert result == NoveltyClaimScientificStructure()
    assert reasons == ()
