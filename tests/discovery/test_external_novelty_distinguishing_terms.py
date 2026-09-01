from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaim,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)


def test_distinguishing_terms_are_backward_compatible():
    draft = NoveltyClaimDraft(
        local_id="c1",
        kind="moderator_interaction",
        text="X depends on Y across wavelength.",
        rationale="test",
    )

    assert draft.distinguishing_terms == []

    claim = NoveltyClaim(
        claim_id="claim:c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text="X depends on Y across wavelength.",
        rationale="test",
    )

    assert claim.distinguishing_terms == []


def test_prompt_requires_specific_distinguishing_facets():
    assert "DISTINGUISHING-FACET CONTRACT:" in _DECOMPOSE_SYSTEM
    assert '"excitation wavelength" or "laser power"' in _DECOMPOSE_SYSTEM
    assert (
        "Do not replace a specific moderator with an umbrella phrase"
        in _DECOMPOSE_SYSTEM
    )
