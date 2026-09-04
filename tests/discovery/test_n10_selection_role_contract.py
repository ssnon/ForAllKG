import pytest
from pydantic import ValidationError

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaim,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)


_ALLOWED = [
    "NOVELTY_BEARING",
    "REQUIRED_ENABLING_RELATION",
    "TESTING_PREDICTION",
    "AUXILIARY",
]


def _draft(**updates):
    payload = {
        "local_id": "c1",
        "kind": "moderator_interaction",
        "importance": "core",
        "text": (
            "Factor Z moderates the association "
            "between descriptor X and outcome Y."
        ),
        "rationale": "Synthetic role-contract control.",
    }
    payload.update(updates)
    return NoveltyClaimDraft(**payload)


def _claim(**updates):
    payload = {
        "claim_id": "claim:1",
        "hypothesis_id": "hypothesis:1",
        "claim_rank": 1,
        "kind": "moderator_interaction",
        "importance": "core",
        "text": (
            "Factor Z moderates the association "
            "between descriptor X and outcome Y."
        ),
        "rationale": "Synthetic role-contract control.",
    }
    payload.update(updates)
    return NoveltyClaim(**payload)


def test_selection_role_is_optional_for_legacy_artifacts():
    assert _draft().novelty_selection_role is None
    assert _claim().novelty_selection_role is None


@pytest.mark.parametrize(
    "role",
    _ALLOWED,
)
def test_selection_role_accepts_declared_vocabulary(role):
    assert (
        _draft(
            novelty_selection_role=role,
        ).novelty_selection_role
        == role
    )

    assert (
        _claim(
            novelty_selection_role=role,
        ).novelty_selection_role
        == role
    )


def test_selection_role_rejects_unknown_value():
    with pytest.raises(ValidationError):
        _draft(
            novelty_selection_role="KNOWN_BACKGROUND",
        )


def test_prompt_makes_selection_role_outcome_blind():
    text = _DECOMPOSE_SYSTEM

    assert "NOVELTY SELECTION-ROLE CONTRACT:" in text

    assert (
        "This role is assigned from the supplied "
        "hypothesis structure only."
        in text
    )

    assert (
        "It is NOT a novelty verdict, prior-art verdict, "
        "truth judgment, or non-obviousness judgment."
        in text
    )

    assert (
        "REQUIRED_ENABLING_RELATION"
        in text
    )

    assert (
        "TESTING_PREDICTION"
        in text
    )

    assert (
        "downstream production behavior remains governed "
        "by the existing importance contract"
        in text
    )
