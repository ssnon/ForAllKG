from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)


def test_decomposition_prompt_requires_typed_atomic_structure():
    text = _DECOMPOSE_SYSTEM

    assert "ATOMIC SCIENTIFIC-STRUCTURE CONTRACT:" in text
    assert "scientific_structure.basis" in text
    assert "exact contiguous span" in text
    assert "introduces_threshold=true" in text
    assert "introduces_regime_change=true" in text
    assert "NEW_REGIME_STRUCTURE" in text
    assert "COUNTER_TO_BASELINE" in text
    assert "DISCRIMINATING_SIGNATURE" in text


def test_prompt_forbids_cross_branch_structure_attribution():
    text = _DECOMPOSE_SYSTEM

    assert (
        "Do not use a wavelength statement as basis for a "
        "laser-power structure"
        in text
    )
    assert (
        "If no valid branch-specific source span exists"
        in text
    )
