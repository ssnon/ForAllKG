from __future__ import annotations

from dac_her.external_novelty_llm import _REVIEW_SYSTEM
from campaigns.sers_standard2.claim_review_dev_validation_v2 import (
    REQUIRED_PROMPT_SENTINELS,
    validate_hardened_prompt,
)


def test_relation_nucleus_prompt_sentinels_present():
    validate_hardened_prompt()
    for sentinel in REQUIRED_PROMPT_SENTINELS:
        assert sentinel in _REVIEW_SYSTEM


def test_legacy_broad_partial_definition_removed():
    assert (
        "a substantial subset or a closely neighboring relation"
        not in _REVIEW_SYSTEM
    )


def test_partial_requires_relation_nucleus_not_thematic_overlap():
    assert (
        "A thematically neighboring relation is not, by itself, "
        "PARTIAL_PRIOR_ART."
        in _REVIEW_SYSTEM
    )
    assert (
        "Evidence from only one context is COMPONENT_ONLY."
        in _REVIEW_SYSTEM
    )
    assert (
        "Evidence for only one arm of the comparison or only the "
        "dependent variable is COMPONENT_ONLY."
        in _REVIEW_SYSTEM
    )


def test_self_consistency_guard_is_present():
    assert "SELF-CONSISTENCY CHECK:" in _REVIEW_SYSTEM
    assert (
        'does not compare X versus Y'
        in _REVIEW_SYSTEM
    )
    assert (
        "PARTIAL_PRIOR_ART is usually inconsistent"
        in _REVIEW_SYSTEM
    )
