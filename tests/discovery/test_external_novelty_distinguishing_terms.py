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
    assert draft.prior_art_identity_terms == []
    assert draft.relation_nucleus_terms == []

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
    assert claim.prior_art_identity_terms == []
    assert claim.relation_nucleus_terms == []


def test_prompt_requires_specific_distinguishing_facets():
    assert "DISTINGUISHING-FACET CONTRACT:" in _DECOMPOSE_SYSTEM
    assert '"excitation wavelength" or "laser power"' in _DECOMPOSE_SYSTEM
    assert (
        "Do not replace a specific moderator with an umbrella phrase"
        in _DECOMPOSE_SYSTEM
    )


def test_prompt_separates_identity_facet_and_relation_nucleus():
    assert (
        "PRIOR-ART IDENTITY / RELATION-NUCLEUS CONTRACT:"
        in _DECOMPOSE_SYSTEM
    )
    assert "prior_art_identity_terms" in _DECOMPOSE_SYSTEM
    assert "relation_nucleus_terms" in _DECOMPOSE_SYSTEM
    assert (
        "A prior-art memory match only makes a historical work "
        "eligible for re-review"
        in _DECOMPOSE_SYSTEM
    )
