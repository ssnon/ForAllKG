from __future__ import annotations

from pipeline_core.discovery.reframing.scientific_synthesis_bounded_authority import (
    _stable_id,
)


def test_bounded_authority_ids_are_deterministic():
    assert _stable_id("x", "a", 1) == _stable_id("x", "a", 1)


def test_bounded_authority_ids_change_with_depth_lineage():
    assert _stable_id("x", "a", 0) != _stable_id("x", "a", 1)
