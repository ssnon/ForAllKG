from __future__ import annotations

import pytest

from pipeline_core.discovery.relation_component_composition import (
    MediatorEquivalenceWitness,
    RelationComponentAuthority,
    RelationComponentProvenance,
    RelationComponentView,
    compose_relation_component_topologies,
    topology_to_task_bridge_composite,
)
from scripts.discovery.build_task_conditioned_axis_plan import (
    _merge_mediator_equivalences,
    _profile_mediator_equivalences,
)


def _candidate_component(
    *,
    component_id: str,
    unit_id: str,
    subject: str,
    relation: str,
    obj: str,
) -> RelationComponentView:
    return RelationComponentView(
        component_id=component_id,
        label=component_id,
        subject=subject,
        relation=relation,
        object=obj,
        authority=RelationComponentAuthority.CANDIDATE_INSPIRATION,
        provenance=RelationComponentProvenance(
            source_kind="candidate_unit",
            source_id=unit_id,
            source_path_id=f"path:{unit_id}",
        ),
        candidate_unit_id=unit_id,
        candidate_unit_score=0.9,
        exploration_score=0.9,
    )


def _sers_pair():
    source = _candidate_component(
        component_id="component:source",
        unit_id="unit:source",
        subject="alpha source",
        relation="is associated with",
        obj="surface-enhanced Raman scattering",
    )
    target = _candidate_component(
        component_id="component:target",
        unit_id="unit:target",
        subject="SERS",
        relation="is associated with",
        obj="beta target",
    )
    return source, target


def test_s25e_disjoint_mediator_aliases_do_not_compose_without_witness():
    source, target = _sers_pair()

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="alpha source",
        requested_target="beta target",
        require_endpoint_fidelity=True,
    )

    assert rows == ()


def test_s25e_explicit_mediator_witness_enables_topology_without_lowering_lexical_gate():
    source, target = _sers_pair()

    witness = MediatorEquivalenceWitness(
        witness_id="mediator:eq:sers",
        left_mediator="surface-enhanced Raman scattering",
        right_mediator="SERS",
        canonical_mediator="SERS",
        witness_kind="explicit_alias",
        provenance_ids=["test:curated_alias:1"],
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="alpha source",
        requested_target="beta target",
        mediator_equivalences=[witness],
        require_endpoint_fidelity=True,
    )

    assert len(rows) == 1
    topology = rows[0]
    assert topology.shared_mediator_tokens == ["sers"]
    assert (
        "mediator_equivalence_witness:mediator:eq:sers"
        in topology.reason_codes
    )
    assert "mediator_argument_slots_equivalent" in topology.reason_codes

    composite = topology_to_task_bridge_composite(topology)
    assert composite.shared_mediator_tokens == ["sers"]


def test_s25e_profile_resolution_aliases_are_automatic_and_auditable():
    witnesses = _profile_mediator_equivalences("sers_au_ag")

    matches = [
        row
        for row in witnesses
        if {
            row.left_mediator.casefold(),
            row.right_mediator.casefold(),
        }
        == {
            "surface-enhanced raman scattering",
            "sers",
        }
    ]

    assert len(matches) == 1
    witness = matches[0]
    assert witness.witness_kind == "domain_profile_normalization"
    assert witness.canonical_mediator == "sers"
    assert witness.provenance_ids


def test_s25e_lexical_one_token_pseudo_bridge_is_still_rejected_without_witness():
    source = _candidate_component(
        component_id="component:s",
        unit_id="unit:s",
        subject="alpha source",
        relation="relates to",
        obj="size",
    )
    target = _candidate_component(
        component_id="component:t",
        unit_id="unit:t",
        subject="particle size",
        relation="relates to",
        obj="beta target",
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="alpha source",
        requested_target="beta target",
        require_endpoint_fidelity=True,
    )

    assert rows == ()


def test_s25e_conflicting_equivalence_for_same_pair_fails_closed():
    left = MediatorEquivalenceWitness(
        witness_id="m1",
        left_mediator="foo",
        right_mediator="bar",
        canonical_mediator="foo",
        witness_kind="explicit_alias",
        provenance_ids=["p1"],
    )
    right = MediatorEquivalenceWitness(
        witness_id="m2",
        left_mediator="bar",
        right_mediator="foo",
        canonical_mediator="baz",
        witness_kind="registry_identity",
        provenance_ids=["p2"],
    )

    with pytest.raises(
        ValueError,
        match="conflicting canonical mediator",
    ):
        _merge_mediator_equivalences(
            (left,),
            (right,),
        )
