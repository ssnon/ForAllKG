from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.reframing.pre_n10_synthesis_bridge import (
    _SYNTHESIS_TYPE_MAP,
    _stable_id,
)


def test_synthesis_kind_map_covers_all_current_kinds():
    assert set(_SYNTHESIS_TYPE_MAP) == {
        "complementary_mechanism_integration",
        "conditional_relation_refinement",
        "competing_model_formulation",
        "measurement_model_integration",
        "cross_lane_explanatory_synthesis",
    }


def test_projection_type_map_uses_only_current_hypothesis_types():
    allowed = {
        "mechanistic_extension",
        "cross_evidence_synthesis",
        "design_lever_interaction",
        "descriptor_mediation",
        "context_dependency",
    }
    assert set(_SYNTHESIS_TYPE_MAP.values()) <= allowed


def test_stable_id_is_deterministic():
    assert _stable_id("x", "a", "b") == _stable_id("x", "a", "b")


def test_stable_id_changes_with_source_identity():
    assert _stable_id("x", "a") != _stable_id("x", "b")
