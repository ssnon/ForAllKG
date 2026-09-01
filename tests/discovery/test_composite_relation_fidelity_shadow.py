from types import SimpleNamespace

from pipeline_core.discovery.composite_relation_fidelity_shadow import (
    CompositeRelationFidelityShadowCritic,
)


def composite_axis():
    return SimpleNamespace(
        axis_id="axis:composite",
        source_mode=(
            "task_conditioned_composite_bridge_projection"
        ),
        proposed_subject=(
            "alpha source signature"
        ),
        proposed_relation=(
            "MAY_RELATE_TO_VIA_COMPOSED_CANDIDATE_BRIDGE"
        ),
        proposed_object=(
            "omega target response"
        ),
        rendered_path=(
            "alpha source signature "
            "-> [UNVERIFIED SOURCE RELATION: source order "
            "| VARIES_WITH | geometry] "
            "-> [SHARED MEDIATOR: geometry, spacing] "
            "-> [UNVERIFIED TARGET RELATION: target response "
            "| VARIES_WITH | geometry] "
            "-> omega target response"
        ),
    )


def card(*observables):
    return SimpleNamespace(
        hypothesis_id="hypothesis:test",
        predicted_observations=[
            SimpleNamespace(
                observable=text
            )
            for text in observables
        ],
    )


def test_complete_conditional_consequence_passes():
    review = (
        CompositeRelationFidelityShadowCritic()
        .review(
            composite_axis(),
            card(
                "Cases with comparable alpha source signature "
                "show distinct omega target response when "
                "geometry or spacing differs."
            ),
        )
    )

    assert review.applicable is True
    assert review.status == "pass"

    assert (
        review
        .observations[0]
        .matched_source_state
        is True
    )

    assert (
        review
        .observations[0]
        .mediator_contrast
        is True
    )

    assert (
        review
        .observations[0]
        .outcome_contrast
        is True
    )


def test_collapse_to_source_readout_fails():
    review = (
        CompositeRelationFidelityShadowCritic()
        .review(
            composite_axis(),
            card(
                "Cases with comparable alpha source signature "
                "but different geometry show non-equivalent "
                "alpha signal strength."
            ),
        )
    )

    assert review.status == "fail"

    assert (
        "no_complete_conditional_consequence"
        in review.reason_codes
    )


def test_split_logic_across_two_observations_fails():
    review = (
        CompositeRelationFidelityShadowCritic()
        .review(
            composite_axis(),
            card(
                "Comparable alpha source cases with different "
                "geometry are evaluated for omega target response.",
                "Omega target response depends on geometry "
                "rather than being invariant.",
            ),
        )
    )

    assert review.status == "fail"

    assert not any(
        row.complete_conditional_consequence
        for row in review.observations
    )


def test_missing_shared_mediator_fails():
    axis = composite_axis()

    axis = SimpleNamespace(
        **{
            **axis.__dict__,
            "rendered_path":
                "alpha -> omega",
        }
    )

    review = (
        CompositeRelationFidelityShadowCritic()
        .review(
            axis,
            card(
                "Comparable alpha cases show different "
                "omega response."
            ),
        )
    )

    assert review.applicable is True
    assert review.status == "fail"

    assert (
        review.reason_codes
        == (
            "shared_mediator_not_recoverable",
        )
    )


def test_non_composite_axis_is_not_applicable():
    axis = composite_axis()

    axis = SimpleNamespace(
        **{
            **axis.__dict__,
            "source_mode":
                "exploratory",
        }
    )

    review = (
        CompositeRelationFidelityShadowCritic()
        .review(
            axis,
            card(
                "Anything."
            ),
        )
    )

    assert review.applicable is False
    assert review.status == "not_applicable"


def test_outcome_contrast_cannot_be_borrowed_from_mediator():
    review = (
        CompositeRelationFidelityShadowCritic()
        .review(
            composite_axis(),
            card(
                "Comparable alpha source cases with different "
                "geometry are evaluated for omega target response."
            ),
        )
    )

    row = review.observations[0]

    assert row.matched_source_state is True
    assert row.mediator_contrast is True

    # "different" qualifies geometry, not the outcome.
    assert row.outcome_contrast is False

    assert review.status == "fail"
