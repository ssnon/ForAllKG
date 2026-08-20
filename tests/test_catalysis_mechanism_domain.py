from __future__ import annotations

from types import SimpleNamespace

import pytest

from domains.catalysis_mechanism.profile import CATALYSIS_MECHANISM_PROFILE
from domains.catalysis_mechanism.extraction import (
    CATALYSIS_MECHANISM_EXTRACTION_ADAPTER,
)


def test_broad_profile_declares_mechanism_first_semantics():
    profile = CATALYSIS_MECHANISM_PROFILE

    assert profile.profile_id == "catalysis_mechanism"
    assert profile.extraction_adapter_id == "catalysis_mechanism"
    assert profile.graph_adapter_id == "catalysis_mechanism"
    assert profile.feasibility_adapter_id is None
    assert "ActiveSite" in profile.resolution.resolvable_node_types
    assert "StructuralState" in profile.resolution.resolvable_node_types
    assert "AdsorbateState" in profile.resolution.resolvable_node_types
    assert "InterfacialEnvironment" in profile.resolution.resolvable_node_types
    assert "MechanisticFactor" in profile.resolution.resolvable_node_types
    assert "Descriptor" in profile.resolution.resolvable_node_types
    assert "CHANGES_RDS" in profile.discovery.mechanism_relation_markers
    assert "FAILS_WHEN" in profile.discovery.mechanism_relation_markers


def test_broad_adapter_accepts_new_mechanism_vocabulary():
    draft = SimpleNamespace(
        entities=[
            SimpleNamespace(type="Catalyst"),
            SimpleNamespace(type="ActiveSite"),
            SimpleNamespace(type="AdsorbateState"),
            SimpleNamespace(type="InterfacialEnvironment"),
            SimpleNamespace(type="MechanisticFactor"),
            SimpleNamespace(type="ReactionStep"),
        ],
        edges=[
            SimpleNamespace(relation="HAS_ACTIVE_SITE"),
            SimpleNamespace(relation="INDUCES"),
            SimpleNamespace(relation="MODULATES"),
            SimpleNamespace(relation="CHANGES_RDS"),
            SimpleNamespace(relation="FAILS_WHEN"),
        ],
    )

    CATALYSIS_MECHANISM_EXTRACTION_ADAPTER.validate_draft_vocabulary(draft)


def test_broad_adapter_fails_closed_on_unregistered_vocabulary():
    invalid_entity = SimpleNamespace(
        entities=[SimpleNamespace(type="MagicMechanism")],
        edges=[],
    )
    with pytest.raises(ValueError, match="MagicMechanism"):
        CATALYSIS_MECHANISM_EXTRACTION_ADAPTER.validate_draft_vocabulary(
            invalid_entity
        )

    invalid_relation = SimpleNamespace(
        entities=[SimpleNamespace(type="Catalyst")],
        edges=[SimpleNamespace(relation="MAKES_BETTER")],
    )
    with pytest.raises(ValueError, match="MAKES_BETTER"):
        CATALYSIS_MECHANISM_EXTRACTION_ADAPTER.validate_draft_vocabulary(
            invalid_relation
        )
