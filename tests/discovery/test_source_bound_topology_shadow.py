from __future__ import annotations

from pipeline_core.discovery.source_bound_topology_shadow import (
    component_closure,
    residual_state,
    source_bound_recoverable_components,
)


def test_source_bound_recovery_requires_exact_proposition_containment():
    composite = {
        "claim_id": "c",
        "kind": "composite",
        "higher_order_relation_basis": [
            "Changing bilayer count changes coupling, and the coupling changes Raman intensity."
        ],
    }
    siblings = [
        {"claim_id": "a", "kind": "mechanistic_link"},
        {"claim_id": "b", "kind": "mechanistic_link"},
    ]
    bindings = {
        "a": {
            "proposition_basis":
                "Changing bilayer count changes coupling"
        },
        "b": {
            "proposition_basis":
                "Ag nanoparticles independently improve reproducibility"
        },
    }
    assert source_bound_recoverable_components(
        composite=composite,
        siblings=siblings,
        binding_by_claim=bindings,
    ) == ["a"]


def test_no_topology_is_not_unsaturated_base():
    state = residual_state(
        full_relation_status="COMPONENTS_ONLY",
        topology_state="NO_COMPONENT_TOPOLOGY",
        component_ids=[],
        backed_component_ids=[],
        closure="NOT_APPLICABLE",
    )
    assert state == "NOT_ASSESSABLE_NO_COMPONENT_TOPOLOGY"


def test_full_relation_and_component_saturation_are_independent():
    state = residual_state(
        full_relation_status="DIRECT_PRIOR_ART",
        topology_state="EXPLICIT",
        component_ids=["a", "b"],
        backed_component_ids=[],
        closure="NONE",
    )
    assert state == "NO_RESIDUAL_FULL_RELATION_BACKED"


def test_same_work_closure_requires_all_components_and_intersection():
    sat = {
        "a": {
            "saturation_claim_status": "PARTIAL_PRIOR_ART",
            "relation_backed_work_ids": ["w1", "w2"],
        },
        "b": {
            "saturation_claim_status": "DIRECT_PRIOR_ART",
            "relation_backed_work_ids": ["w1"],
        },
    }
    closure, backed, unresolved, shared = component_closure(
        ["a", "b"], sat
    )
    assert closure == "SAME_WORK"
    assert backed == ["a", "b"]
    assert unresolved == []
    assert shared == ["w1"]
