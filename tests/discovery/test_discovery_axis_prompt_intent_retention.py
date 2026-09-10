from __future__ import annotations

from pipeline_core.discovery.discovery_axis_prompt import (
    FAMILY_AWARE_PROMPT_VERSION,
    PROMPT_VERSION,
    _SYSTEM_APPENDIX,
)


def _normalized_prompt() -> str:
    return " ".join(_SYSTEM_APPENDIX.split())


def test_discovery_axis_prompt_preserves_assigned_relation_roles() -> None:
    prompt = _normalized_prompt()
    assert "DISCOVERY-INTENT RETENTION" in prompt
    assert "proposed_subject" in prompt
    assert "proposed_relation" in prompt
    assert "proposed_object" in prompt
    assert (
        "inferential_bridge must make that same assigned dependency the primary"
        in prompt
    )
    assert (
        "one predicted observation and one matching falsification criterion"
        in prompt
    )
    assert (
        "they may not replace the assigned axis as the central hypothesis relation"
        in prompt
    )
    assert "If the assigned relation cannot be preserved" in prompt


def test_discovery_intent_policy_remains_bounded_and_domain_neutral() -> None:
    prompt = _normalized_prompt()
    assert "inspiration-only" in prompt
    assert "MUST NOT become a positive premise" in prompt
    assert "does not authorize" in prompt
    assert "external novelty claim" in prompt

    # S17 treatment wording and DAC/HER-specific fallback examples are not
    # production prompt policy.
    assert "experimental comparison arm" not in prompt
    assert "CLEAN-CYCLE" not in prompt
    assert "coordination→adsorption→HER" not in prompt
    assert "hydrogen adsorption free energy" not in prompt

    assert PROMPT_VERSION.endswith("-s17-intent-v1")
    assert FAMILY_AWARE_PROMPT_VERSION.endswith("-s17-intent-v1")
