from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.higher_order_topology_carrier import (
    HigherOrderSynthesisCarrier,
    build_topology_native_synthesis_carriers,
    topology_native_synthesis_carrier,
)
from pipeline_core.discovery.higher_order_topology_composition import (
    EligibleModifierComponent,
    ModifierEligibilityWitness,
    compose_higher_order_topologies,
)
from pipeline_core.discovery.relation_component_composition import (
    EndpointEquivalenceWitness,
    RelationComponentAuthority,
    RelationComponentProvenance,
    RelationComponentView,
    compose_relation_component_topologies,
)


def _known(
    *,
    node_id: str,
    subject: str,
    relation: str,
    object_: str,
) -> RelationComponentView:
    return RelationComponentView(
        component_id="component:" + node_id,
        label=node_id,
        subject=subject,
        relation=relation,
        object=object_,
        authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        provenance=RelationComponentProvenance(
            source_kind="accepted_pattern",
            source_id=node_id,
            paper_id="paper:test",
            chunk_id="chunk:test",
            document_id="document:test",
        ),
        accepted_pattern_id=node_id,
    )


def _higher_order():
    source = _known(
        node_id="known:distance-field",
        subject="interparticle distance",
        relation="MODULATES",
        object_="electric field enhancement",
    )
    target = _known(
        node_id="known:field-signal",
        subject="electric field enhancement",
        relation="PROMOTES",
        object_="SERS signal",
    )

    witnesses = (
        EndpointEquivalenceWitness(
            witness_id="eq:source",
            left_endpoint="interparticle separation",
            right_endpoint="interparticle distance",
            witness_kind="task_supplied",
            provenance_ids=["review:test"],
        ),
        EndpointEquivalenceWitness(
            witness_id="eq:target",
            left_endpoint="SERS intensity",
            right_endpoint="SERS signal",
            witness_kind="task_supplied",
            provenance_ids=["review:test"],
        ),
    )

    backbone = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        endpoint_equivalences=witnesses,
        require_endpoint_fidelity=True,
    )[0]

    modifier = _known(
        node_id="known:roughness-field",
        subject="surface roughness",
        relation="MODULATES",
        object_="electric field enhancement",
    )

    eligible = EligibleModifierComponent(
        component=modifier,
        eligibility=ModifierEligibilityWitness(
            witness_id="modifier-witness:roughness-field",
            modifier_component_id=modifier.component_id,
            validated_anchor_role="mediator",
            validated_anchor_slot="object",
            modifier_slot="subject",
            validated_anchor_text="electric field enhancement",
            modifier_text="surface roughness",
            anchor_purity_pass=True,
            provenance_ids=["evaluation:s23c:test"],
        ),
    )

    return compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[eligible],
    )[0]


def _carrier():
    return topology_native_synthesis_carrier(
        topology=_higher_order(),
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )


def test_carrier_preserves_topology_component_and_provenance_lineage():
    topology = _higher_order()
    carrier = topology_native_synthesis_carrier(
        topology=topology,
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )

    assert carrier.topology.topology_id == topology.topology_id
    assert (
        carrier.lineage.backbone_topology_id
        == topology.backbone_topology_id
    )
    assert (
        carrier.lineage.source_component_id
        == topology.backbone.source_component.component_id
    )
    assert (
        carrier.lineage.target_component_id
        == topology.backbone.target_component.component_id
    )
    assert (
        carrier.lineage.modifier_component_id
        == topology.modifier_component.component_id
    )
    assert (
        carrier.lineage.source_provenance_source_id
        == topology.backbone.source_component.provenance.source_id
    )
    assert (
        carrier.lineage.target_provenance_source_id
        == topology.backbone.target_component.provenance.source_id
    )
    assert (
        carrier.lineage.modifier_provenance_source_id
        == topology.modifier_component.provenance.source_id
    )


def test_carrier_preserves_endpoint_and_modifier_witness_authority():
    topology = _higher_order()
    carrier = topology_native_synthesis_carrier(
        topology=topology,
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )

    assert carrier.lineage.source_endpoint_binding_authority == "equivalent"
    assert carrier.lineage.target_endpoint_binding_authority == "equivalent"
    assert (
        carrier.lineage.source_endpoint_equivalence_witness_id
        == "eq:source"
    )
    assert (
        carrier.lineage.target_endpoint_equivalence_witness_id
        == "eq:target"
    )
    assert (
        carrier.lineage.modifier_eligibility_witness_id
        == topology.modifier_eligibility.witness_id
    )
    assert carrier.lineage.modifier_anchor_role == "mediator"
    assert carrier.lineage.modifier_anchor_slot == "object"
    assert carrier.lineage.modifier_slot == "subject"


def test_carrier_cannot_gain_axis_candidate_or_novelty_authority():
    carrier = _carrier()

    assert carrier.epistemic_status == "inspiration_only"
    assert carrier.requires_verification is True
    assert carrier.novelty_authority is False
    assert carrier.shadow_only is True
    assert carrier.discovery_axis_materialization_authorized is False
    assert carrier.candidate_anchor_synthesized is False

    with pytest.raises(ValidationError):
        HigherOrderSynthesisCarrier.model_validate(
            {
                **carrier.model_dump(),
                "novelty_authority": True,
            }
        )

    with pytest.raises(ValidationError):
        HigherOrderSynthesisCarrier.model_validate(
            {
                **carrier.model_dump(),
                "candidate_anchor_synthesized": True,
            }
        )


def test_carrier_lineage_tampering_fails_closed():
    carrier = _carrier()
    payload = carrier.model_dump()
    payload["lineage"]["modifier_component_id"] = "component:wrong"

    with pytest.raises(
        ValidationError,
        match="modifier_component_id",
    ):
        HigherOrderSynthesisCarrier.model_validate(payload)


def test_carrier_id_is_deterministic():
    topology = _higher_order()

    a = topology_native_synthesis_carrier(
        topology=topology,
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )
    b = topology_native_synthesis_carrier(
        topology=topology,
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )

    assert a.carrier_id == b.carrier_id


def test_batch_projection_is_one_to_one_and_preserves_order():
    topology = _higher_order()

    rows = build_topology_native_synthesis_carriers(
        topologies=[topology],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )

    assert len(rows) == 1
    assert rows[0].topology.topology_id == topology.topology_id


def test_duplicate_topology_ids_fail_closed_instead_of_silent_dedup():
    topology = _higher_order()

    with pytest.raises(
        ValueError,
        match="duplicate higher-order topology id",
    ):
        build_topology_native_synthesis_carriers(
            topologies=[topology, topology],
            requested_source="interparticle separation",
            requested_target="SERS intensity",
        )
