from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
    compose_task_bridge_candidates,
    diagnose_task_bridge_candidate,
)


def _candidate(
    unit_id: str,
    subject: str,
    relation: str,
    object_: str,
) -> CandidateRelationView:
    return CandidateRelationView(
        unit_id=unit_id,
        label=unit_id,
        proposed_subject=subject,
        proposed_relation=relation,
        proposed_object=object_,
    )


def test_raman_legacy_bridge_is_partial_and_role_incompatible():
    source = _candidate(
        "candidate_unit:source",
        "molecular adsorption affinity",
        "VARIES_WITH",
        "surface identity",
    )
    target = _candidate(
        "candidate_unit:target",
        "surface-plasmon multimode intensity",
        "VARIES_WITH",
        "nanopore geometry",
    )

    rows = compose_task_bridge_candidates(
        candidates=[source, target],
        requested_source="molecular orientation",
        requested_target="Raman intensity",
    )

    assert len(rows) == 1
    assert rows[0].source_overlap_tokens == [
        "molecular"
    ]
    assert rows[0].target_overlap_tokens == [
        "intensity"
    ]
    assert rows[0].shared_mediator_tokens == [
        "surface"
    ]

    shadow = diagnose_task_bridge_candidate(
        composite=rows[0],
        requested_source="molecular orientation",
        requested_target="Raman intensity",
    )

    assert (
        shadow["source_endpoint_binding"]
        ["binding_authority"]
        == "partial"
    )
    assert (
        shadow["source_endpoint_binding"]
        ["binding_slot"]
        == "subject"
    )
    assert (
        shadow["source_endpoint_binding"]
        ["task_coverage"]
        == 0.5
    )

    assert (
        shadow["target_endpoint_binding"]
        ["binding_authority"]
        == "partial"
    )
    assert (
        shadow["target_endpoint_binding"]
        ["binding_slot"]
        == "subject"
    )
    assert (
        shadow["target_endpoint_binding"]
        ["task_coverage"]
        == 0.5
    )

    assert (
        shadow["role_aware_source_mediator_tokens"]
        == ["identity", "surface"]
    )
    assert (
        shadow["role_aware_target_mediator_tokens"]
        == ["geometry", "nanopore"]
    )
    assert (
        shadow["role_aware_shared_mediator_tokens"]
        == []
    )
    assert not shadow[
        "role_aware_mediator_compatible"
    ]
    assert not shadow[
        "exact_endpoint_fidelity"
    ]
    assert not shadow[
        "strict_materializable_without_equivalence_witness"
    ]
    assert shadow["shadow_only"] is True
    assert (
        shadow["production_selection_changed"]
        is False
    )


def test_exact_slot_endpoints_and_opposite_slot_mediator_are_strict():
    source = _candidate(
        "source",
        "molecular orientation",
        "VARIES_WITH",
        "surface field",
    )
    target = _candidate(
        "target",
        "Raman intensity",
        "VARIES_WITH",
        "surface field",
    )

    rows = compose_task_bridge_candidates(
        candidates=[source, target],
        requested_source="molecular orientation",
        requested_target="Raman intensity",
    )

    assert len(rows) == 1

    shadow = diagnose_task_bridge_candidate(
        composite=rows[0],
        requested_source="molecular orientation",
        requested_target="Raman intensity",
    )

    assert (
        shadow["source_endpoint_binding"]
        ["binding_authority"]
        == "exact"
    )
    assert (
        shadow["target_endpoint_binding"]
        ["binding_authority"]
        == "exact"
    )
    assert shadow[
        "role_aware_shared_mediator_tokens"
    ] == ["field", "surface"]
    assert shadow[
        "role_aware_mediator_compatible"
    ]
    assert shadow[
        "exact_endpoint_fidelity"
    ]
    assert shadow[
        "strict_materializable_without_equivalence_witness"
    ]


def test_e2e_stage75_wires_shadow_artifact_without_authority_change():
    source = Path(
        "scripts/discovery/"
        "run_dac_discovery_e2e.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "hypothesis_axis_a4."
        "task_bridge_fidelity.shadow.json"
        in source
    )
    assert (
        '"--output-legacy-bridge-shadow"'
        in source
    )
    assert (
        "task_bridge_fidelity_shadow,"
        in source
    )

    builder = Path(
        "scripts/discovery/"
        "build_task_conditioned_axis_plan.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        '"production_selection_changed": False'
        in builder
    )
    assert (
        '"shadow_only": True'
        in builder
    )
