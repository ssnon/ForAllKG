import pytest
from pydantic import ValidationError

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)


def _claim(
    local_id,
    *,
    importance,
):
    return NoveltyClaimDraft(
        local_id=local_id,
        kind="distinctive_prediction",
        importance=importance,
        text="A testable atomic scientific relation.",
        rationale="Synthetic contract control.",
    )


def test_new_decomposition_requires_at_least_one_core_claim():
    with pytest.raises(
        ValidationError,
        match="at least one core claim",
    ):
        NoveltyClaimDecompositionDraft(
            claims=[
                _claim(
                    "c1",
                    importance="supporting",
                ),
                _claim(
                    "c2",
                    importance="supporting",
                ),
            ]
        )


def test_core_plus_supporting_claims_are_valid():
    draft = NoveltyClaimDecompositionDraft(
        claims=[
            _claim(
                "c1",
                importance="core",
            ),
            _claim(
                "c2",
                importance="supporting",
            ),
        ]
    )

    assert [
        row.importance
        for row in draft.claims
    ] == [
        "core",
        "supporting",
    ]


def test_prompt_defines_core_selection_role():
    text = _DECOMPOSE_SYSTEM

    assert "CLAIM IMPORTANCE CONTRACT:" in text
    assert (
        "Every decomposition MUST contain at least one "
        "importance=core claim."
        in text
    )
    assert (
        "One apparently stronger branch must not hide "
        "another central branch."
        in text
    )
    assert (
        "Do not mark every claim supporting"
        in text
    )
