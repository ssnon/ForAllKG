import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)


def test_non_composite_novelty_claim_cannot_declare_higher_order_components():
    with pytest.raises(ValueError):
        NoveltyClaimDecompositionDraft(
            claims=[
                NoveltyClaimDraft(
                    local_id="c1",
                    kind="mechanistic_link",
                    text="A affects B.",
                    rationale="atomic relation",
                    higher_order_component_local_ids=["c2"],
                )
            ]
        )


def test_composite_novelty_claim_may_declare_known_local_components():
    draft = NoveltyClaimDecompositionDraft(
        claims=[
            NoveltyClaimDraft(
                local_id="c1",
                kind="mechanistic_link",
                text="A affects B.",
                rationale="component 1",
            ),
            NoveltyClaimDraft(
                local_id="c2",
                kind="mechanistic_link",
                text="B affects C.",
                rationale="component 2",
            ),
            NoveltyClaimDraft(
                local_id="c3",
                kind="composite",
                text="A affects C through B.",
                rationale="explicit composite",
                higher_order_component_local_ids=["c1", "c2"],
            ),
        ]
    )
    assert draft.claims[-1].higher_order_component_local_ids == ["c1", "c2"]
