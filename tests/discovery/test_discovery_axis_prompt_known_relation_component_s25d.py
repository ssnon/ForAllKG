from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.discovery_axis_prompt import (
    _s25d_known_relation_component_guidance,
)


def _axis(required: bool):
    return SimpleNamespace(
        second_order_gap_required=required,
        inspiration_role=(
            "KNOWN_RELATION_COMPONENT"
            if required
            else "EXPLORATORY_AXIS"
        ),
        external_relation_source_mode=(
            "SOURCE_REPORTED"
            if required
            else "BOUNDED_SYNTHESIS"
        ),
        proposed_subject="A",
        proposed_relation="affects",
        proposed_object="B",
    )


def test_s25d_known_component_guidance_demands_second_order_gap():
    text = _s25d_known_relation_component_guidance(
        _axis(True)
    )

    assert "KNOWN RELATION COMPONENT MODE" in text
    assert "MODERATOR" in text
    assert "INTERACTION" in text
    assert "RESIDUAL" in text
    assert "BOUNDARY" in text
    assert "PROXY_DECOUPLING" in text
    assert "COMPENSATION_LIMIT" in text
    assert "positive premise" in text


def test_s25d_exploratory_axis_has_no_special_guidance():
    assert _s25d_known_relation_component_guidance(
        _axis(False)
    ) == ""
