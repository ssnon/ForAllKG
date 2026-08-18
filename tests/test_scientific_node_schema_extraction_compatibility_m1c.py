from __future__ import annotations

import dac_her.schemas as legacy
import pipeline_core.scientific_node_schema as core


def test_legacy_scientific_leaf_symbols_are_core_objects():
    assert legacy.EntityType is core.EntityType
    assert legacy.CalculationType is core.CalculationType
    assert legacy.ObservationClaimType is core.ObservationClaimType
    assert legacy.MechanismClaimType is core.MechanismClaimType
    assert legacy.MechanismBasis is core.MechanismBasis

    assert legacy.EntityNode is core.EntityNode
    assert legacy.CalculationNode is core.CalculationNode
    assert legacy.ObservationClaimNode is core.ObservationClaimNode
    assert legacy.MechanismClaimNode is core.MechanismClaimNode


def test_scientific_leaf_models_are_owned_by_pipeline_core():
    assert core.EntityNode.__module__ == (
        "pipeline_core.scientific_node_schema"
    )
    assert core.CalculationNode.__module__ == (
        "pipeline_core.scientific_node_schema"
    )
    assert core.ObservationClaimNode.__module__ == (
        "pipeline_core.scientific_node_schema"
    )
    assert core.MechanismClaimNode.__module__ == (
        "pipeline_core.scientific_node_schema"
    )


def test_known_domain_vocabularies_remain_legacy_owned():
    assert hasattr(legacy, "KnownEntityType")
    assert hasattr(legacy, "KnownCalculationType")
    assert hasattr(legacy, "KnownObservationClaimType")
    assert hasattr(legacy, "KnownMechanismClaimType")

    assert not hasattr(core, "KnownEntityType")
    assert not hasattr(core, "KnownCalculationType")
    assert not hasattr(core, "KnownObservationClaimType")
    assert not hasattr(core, "KnownMechanismClaimType")


def test_experiment_node_is_outside_scientific_node_schema():
    assert not hasattr(core, "ExperimentNode")
