from pipeline_core.discovery.external_novelty_llm import _REVIEW_SYSTEM


def test_conditional_distinctive_prediction_requires_moderator_relation() -> None:
    assert (
        "explicit moderator, context condition, interaction, or "
        "cross-context comparison"
        in _REVIEW_SYSTEM
    )
    assert (
        "WITHOUT that moderator/conditional contrast is COMPONENT_ONLY"
        in _REVIEW_SYSTEM
    )


def test_conditional_partial_prior_art_must_preserve_relation_nucleus() -> None:
    assert (
        "defining conditional/moderating relation or an equivalent contrast "
        "is still represented"
        in _REVIEW_SYSTEM
    )


def test_unconditional_distinctive_prediction_can_still_be_partial() -> None:
    assert (
        "For an unconditional distinctive prediction"
        in _REVIEW_SYSTEM
    )
    assert (
        "may qualify as PARTIAL_PRIOR_ART"
        in _REVIEW_SYSTEM
    )


def test_base_relation_alone_is_not_partial_for_conditional_claim() -> None:
    assert (
        "only the unconditioned base relation of a conditional claim "
        "is COMPONENT_ONLY"
        in _REVIEW_SYSTEM
    )


def test_old_overpermissive_distinctive_prediction_rule_is_gone() -> None:
    old_rule = (
        "For distinctive_prediction claims, PARTIAL_PRIOR_ART requires "
        "the same dependent relation or contrast even if direction, "
        "material scope, or a control condition is incomplete."
    )
    assert old_rule not in _REVIEW_SYSTEM
