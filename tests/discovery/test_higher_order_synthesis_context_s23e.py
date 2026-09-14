from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.higher_order_synthesis_context import (
    HigherOrderSynthesisContext,
    build_higher_order_synthesis_contexts,
    higher_order_synthesis_context,
    render_higher_order_shadow_prompt,
)
from pipeline_core.discovery.higher_order_topology_carrier import (
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


def _carrier():
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

    topology = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[eligible],
    )[0]

    return topology_native_synthesis_carrier(
        topology=topology,
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )


def test_context_preserves_three_exact_relation_premises():
    carrier = _carrier()
    context = higher_order_synthesis_context(
        carrier=carrier
    )

    assert context.carrier_id == carrier.carrier_id
    assert context.lineage == carrier.lineage

    by_role = {
        row.premise_role: row
        for row in context.premises
    }
    assert set(by_role) == {
        "source_backbone_relation",
        "target_backbone_relation",
        "modifier_relation",
    }

    assert (
        by_role["source_backbone_relation"].subject
        == "interparticle distance"
    )
    assert (
        by_role["source_backbone_relation"].relation
        == "MODULATES"
    )
    assert (
        by_role["target_backbone_relation"].object
        == "SERS signal"
    )
    assert (
        by_role["modifier_relation"].subject
        == "surface roughness"
    )


def test_context_preserves_component_authority_and_provenance():
    context = higher_order_synthesis_context(
        carrier=_carrier()
    )

    for premise in context.premises:
        assert premise.authority == "confirmed_known"
        assert premise.epistemic_use == "confirmed_known_component"
        assert premise.provenance_source_id.startswith("known:")


def test_context_preserves_role_bound_structural_opportunity():
    context = higher_order_synthesis_context(
        carrier=_carrier()
    )
    opportunity = context.structural_opportunity

    assert opportunity.modifier_text == "surface roughness"
    assert opportunity.modifier_anchor_role == "mediator"
    assert opportunity.modifier_anchor_text == "electric field enhancement"
    assert (
        opportunity.source_side_mediator_text
        == "electric field enhancement"
    )
    assert (
        opportunity.target_side_mediator_text
        == "electric field enhancement"
    )


def test_context_cannot_gain_novelty_axis_candidate_or_llm_authority():
    payload = higher_order_synthesis_context(
        carrier=_carrier()
    ).model_dump(mode="json")

    for path, value in (
        (("novelty_authority",), True),
        (
            (
                "guard",
                "discovery_axis_materialization_authorized",
            ),
            True,
        ),
        (
            (
                "guard",
                "candidate_anchor_synthesis_authorized",
            ),
            True,
        ),
        (("guard", "llm_call_authorized"), True),
        (("guard", "mediator_identity_assertion_authorized"), True),
    ):
        candidate = higher_order_synthesis_context(
            carrier=_carrier()
        ).model_dump(mode="json")

        cursor = candidate
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value

        with pytest.raises(ValidationError):
            HigherOrderSynthesisContext.model_validate(
                candidate
            )


def test_context_lineage_tampering_fails_closed():
    payload = higher_order_synthesis_context(
        carrier=_carrier()
    ).model_dump(mode="json")

    payload["premises"][0]["component_id"] = "component:tampered"

    with pytest.raises(
        ValidationError,
        match="lost component lineage",
    ):
        HigherOrderSynthesisContext.model_validate(
            payload
        )


def test_shadow_prompt_separates_premises_from_unverified_interaction():
    prompt = render_higher_order_shadow_prompt(
        higher_order_synthesis_context(
            carrier=_carrier()
        )
    )

    assert (
        "interparticle distance --MODULATES--> "
        "electric field enhancement"
    ) in prompt
    assert (
        "surface roughness --MODULATES--> "
        "electric field enhancement"
    ) in prompt

    assert "NOT evidence of interaction or novelty" in prompt
    assert "as a HYPOTHESIS" in prompt
    assert "Do not reverse or strengthen" in prompt
    assert "not asserted identical entities" in prompt
    assert "mediator path" in prompt
    assert "Keep effect direction open" in prompt
    assert "N10 novelty review" in prompt
    assert "no LLM call is authorized" in prompt


def test_batch_context_projection_is_one_to_one_and_duplicate_fails_closed():
    carrier = _carrier()
    contexts = build_higher_order_synthesis_contexts(
        carriers=[carrier]
    )

    assert len(contexts) == 1
    assert contexts[0].carrier_id == carrier.carrier_id

    with pytest.raises(
        ValueError,
        match="duplicate higher-order synthesis carrier id",
    ):
        build_higher_order_synthesis_contexts(
            carriers=[carrier, carrier]
        )
