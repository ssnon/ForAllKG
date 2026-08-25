from __future__ import annotations

from pipeline_core.discovery.external_novelty_llm import (
    _REVIEW_SYSTEM,
)


def test_moderator_interaction_direct_requires_relation_nucleus() -> None:
    prompt = _REVIEW_SYSTEM

    assert (
        "MODERATOR-INTERACTION DIRECTNESS:"
        in prompt
    )

    assert (
        "DIRECT_PRIOR_ART requires the supplied "
        "title/abstract metadata to explicitly state, "
        "test, compare, or demonstrate that the moderator "
        "changes, conditions, or modifies the base relation itself."
        in prompt
    )

    assert (
        "M changes how X affects Y"
        in prompt
    )

    assert (
        "joint M-by-X effect on Y"
        in prompt
    )


def test_moderator_separate_main_effects_remain_components_only() -> None:
    prompt = _REVIEW_SYSTEM

    assert (
        "If a record separately establishes X affects Y "
        "and M affects Y, classify it as COMPONENT_ONLY."
        in prompt
    )

    assert (
        "adjacent sentences"
        in prompt
    )

    assert (
        "Do not infer moderator interaction from local "
        "textual adjacency"
        in prompt
    )

    assert (
        "If not, use COMPONENT_ONLY."
        in prompt
    )


def test_moderator_partial_still_requires_interaction_nucleus() -> None:
    prompt = _REVIEW_SYSTEM

    assert (
        "PARTIAL_PRIOR_ART still requires the "
        "interaction/conditional relation nucleus to be represented."
        in prompt
    )

    assert (
        "the moderator relation itself may not be inferred."
        in prompt
    )
