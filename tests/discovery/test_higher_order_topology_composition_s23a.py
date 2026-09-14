from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.higher_order_topology_composition import (
    EligibleModifierComponent,
    ModifierEligibilityWitness,
    compose_higher_order_topologies,
)
from pipeline_core.discovery.relation_component_composition import (
    EndpointEquivalenceWitness,
    RelationComponentAuthority,
    candidate_inspiration_component,
    compose_relation_component_topologies,
    confirmed_known_component_from_mapping,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
)


def _accepted(
    *,
    node_id: str,
    subject: str,
    relation: str,
    object_: str,
):
    source_phrase = f"{subject} {relation.lower()} {object_}"

    return {
        "node_id": node_id,
        "source_local_id": node_id,
        "concept_type": "RelationPattern",
        "label": source_phrase,
        "source_phrase": source_phrase,
        "description": "",
        "retention_lane": "accepted_pattern",
        "evidence_scope": "paper_result",
        "pattern_subject": subject,
        "pattern_relation": relation,
        "pattern_object": object_,
        "relation_strength": "causal_interpretive",
        "qualifiers_json": "[]",
        "pattern_support_mode": "explicit_single_span",
        "supporting_phrases_json": json.dumps(
            [source_phrase]
        ),
        "subject_evidence_phrase": subject,
        "relation_evidence_phrase": relation.lower(),
        "object_evidence_phrase": object_,
        "comparison_items_json": "[]",
        "paper_id": "paper:test",
        "chunk_id": "chunk:test",
        "document_id": "document:test",
    }


def _known(
    *,
    node_id: str,
    subject: str,
    relation: str,
    object_: str,
):
    return confirmed_known_component_from_mapping(
        _accepted(
            node_id=node_id,
            subject=subject,
            relation=relation,
            object_=object_,
        )
    )


def _candidate(
    *,
    unit_id: str,
    subject: str,
    relation: str,
    object_: str,
):
    return candidate_inspiration_component(
        relation=CandidateRelationView(
            unit_id=unit_id,
            proposed_subject=subject,
            proposed_relation=relation,
            proposed_object=object_,
        ),
        candidate_unit_score=0.60,
        exploration_score=0.20,
        quality_eligible=True,
        source_path_id=f"path:{unit_id}",
    )


def _eligible(
    component,
    *,
    anchor_role="mediator",
    anchor_slot="object",
    modifier_slot="subject",
):
    anchor_text = (
        component.subject
        if anchor_slot == "subject"
        else component.object
    )
    modifier_text = (
        component.subject
        if modifier_slot == "subject"
        else component.object
    )

    return EligibleModifierComponent(
        component=component,
        eligibility=ModifierEligibilityWitness(
            witness_id=(
                "s21l:"
                + component.component_id
                + ":"
                + anchor_role
            ),
            modifier_component_id=(
                component.component_id
            ),
            validated_anchor_role=anchor_role,
            validated_anchor_slot=anchor_slot,
            modifier_slot=modifier_slot,
            validated_anchor_text=anchor_text,
            modifier_text=modifier_text,
            anchor_purity_pass=True,
            provenance_ids=[
                "evaluation:s21l:test"
            ],
            reason_codes=[
                "anchor_purity_pass",
                "source_leakage_pass",
                "target_leakage_pass",
                "mediator_leakage_pass",
                "genericity_pass",
                "directional_role_pass",
            ],
        ),
    )


def _backbone(
    *,
    strict: bool = True,
):
    source = _known(
        node_id="known:distance-field",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle distance",
    )
    target = _known(
        node_id="known:field-signal",
        subject="electric field enhancement",
        relation="PROMOTES",
        object_="SERS signal",
    )

    witnesses = [
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
    ]

    return compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        endpoint_equivalences=(
            witnesses
            if strict
            else ()
        ),
        require_endpoint_fidelity=strict,
    )[0]


def test_known_modifier_can_attach_to_mediator_without_novelty_authority():
    backbone = _backbone()

    modifier = _known(
        node_id="known:roughness-field",
        subject="surface roughness",
        relation="MODULATES",
        object_="electric field enhancement",
    )

    rows = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[_eligible(modifier)],
    )

    assert len(rows) == 1
    row = rows[0]

    assert (
        row.role_binding.modifier_anchor_role
        == "mediator"
    )
    assert (
        row.role_binding.modifier_text
        == "surface roughness"
    )
    assert (
        row.modifier_authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
    )
    assert row.epistemic_status == "inspiration_only"
    assert row.requires_verification
    assert row.novelty_authority is False
    assert row.shadow_only is True


def test_candidate_modifier_preserves_candidate_authority():
    backbone = _backbone()

    modifier = _candidate(
        unit_id="candidate:wavelength-field",
        subject="excitation wavelength",
        relation="MODULATES",
        object_="electric field enhancement",
    )

    row = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[_eligible(modifier)],
    )[0]

    assert (
        row.modifier_authority
        == RelationComponentAuthority.CANDIDATE_INSPIRATION
    )
    assert (
        row.modifier_component.candidate_unit_id
        == "candidate:wavelength-field"
    )
    assert row.novelty_authority is False


def test_modifier_can_attach_to_source_endpoint_alias():
    backbone = _backbone()

    modifier = _known(
        node_id="known:aggregation-distance",
        subject="aggregation state",
        relation="MODULATES",
        object_="interparticle separation",
    )

    row = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[_eligible(
            modifier,
            anchor_role="source",
            anchor_slot="object",
            modifier_slot="subject",
        )],
    )[0]

    assert (
        row.role_binding.modifier_anchor_role
        == "source"
    )
    assert (
        row.role_binding.modifier_text
        == "aggregation state"
    )


def test_modifier_can_attach_to_target_endpoint_alias():
    backbone = _backbone()

    modifier = _known(
        node_id="known:analyte-sers",
        subject="analyte adsorption",
        relation="MODULATES",
        object_="SERS intensity",
    )

    row = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[_eligible(
            modifier,
            anchor_role="target",
            anchor_slot="object",
            modifier_slot="subject",
        )],
    )[0]

    assert (
        row.role_binding.modifier_anchor_role
        == "target"
    )
    assert (
        row.role_binding.modifier_text
        == "analyte adsorption"
    )


def test_partial_backbone_is_not_eligible_for_higher_order_composition():
    source = _known(
        node_id="known:distance-field",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle distance",
    )
    target = _known(
        node_id="known:field-signal",
        subject="electric field enhancement",
        relation="PROMOTES",
        object_="SERS signal",
    )

    partial_backbone = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )[0]

    modifier = _known(
        node_id="known:roughness-field",
        subject="surface roughness",
        relation="MODULATES",
        object_="electric field enhancement",
    )

    rows = compose_higher_order_topologies(
        backbones=[partial_backbone],
        modifiers=[_eligible(modifier)],
    )

    assert rows == ()


def test_modifier_eligibility_witness_is_bound_to_component():
    modifier = _known(
        node_id="known:roughness-field",
        subject="surface roughness",
        relation="MODULATES",
        object_="electric field enhancement",
    )

    with pytest.raises(
        ValidationError,
        match="component id mismatch",
    ):
        EligibleModifierComponent(
            component=modifier,
            eligibility=ModifierEligibilityWitness(
                witness_id="s21l:wrong",
                modifier_component_id="wrong-component",
                validated_anchor_role="mediator",
                validated_anchor_slot="object",
                modifier_slot="subject",
                validated_anchor_text="electric field enhancement",
                modifier_text="surface roughness",
                anchor_purity_pass=True,
                provenance_ids=["evaluation:s21l:test"],
            ),
        )


def test_modifier_cannot_reuse_backbone_component():
    backbone = _backbone()

    reused = _eligible(
        backbone.source_component
    )

    rows = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[reused],
    )

    assert rows == ()


def test_exact_modifier_leakage_into_backbone_role_is_rejected_defensively():
    backbone = _backbone()

    # Anchor = mediator; opposite slot repeats source endpoint exactly.
    modifier = _known(
        node_id="known:leaky",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle distance",
    )

    rows = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[_eligible(modifier)],
    )

    assert rows == ()


def test_ambiguous_attachment_role_fails_closed():
    backbone = _backbone()

    # The relation spans source and target roles, so either argument can
    # attach to a different backbone role. That is not a modifier C.
    modifier = _known(
        node_id="known:ambiguous",
        subject="interparticle separation",
        relation="VARIES_WITH",
        object_="SERS intensity",
    )

    rows = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[_eligible(modifier)],
    )

    assert rows == ()


def test_composer_does_not_reinterpret_upstream_validated_role():
    backbone = _backbone()

    modifier = _known(
        node_id="known:roughness-field",
        subject="surface roughness",
        relation="MODULATES",
        object_="electric field enhancement",
    )

    rows = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[_eligible(
            modifier,
            anchor_role="source",
            anchor_slot="object",
            modifier_slot="subject",
        )],
    )

    assert rows == ()


def test_modifier_anchor_purity_is_explicit_upstream_authority():
    modifier = _known(
        node_id="known:roughness-field",
        subject="surface roughness",
        relation="MODULATES",
        object_="electric field enhancement",
    )

    with pytest.raises(
        ValidationError,
        match="anchor_purity_pass",
    ):
        ModifierEligibilityWitness(
            witness_id="s21l:impure",
            modifier_component_id=modifier.component_id,
            validated_anchor_role="mediator",
            validated_anchor_slot="object",
            modifier_slot="subject",
            validated_anchor_text="electric field enhancement",
            modifier_text="surface roughness",
            anchor_purity_pass=False,
            provenance_ids=["evaluation:s21l:test"],
        )


def test_modifier_witness_slot_text_mismatch_fails_closed():
    modifier = _known(
        node_id="known:roughness-field",
        subject="surface roughness",
        relation="MODULATES",
        object_="electric field enhancement",
    )

    with pytest.raises(
        ValidationError,
        match="anchor text/slot mismatch",
    ):
        EligibleModifierComponent(
            component=modifier,
            eligibility=ModifierEligibilityWitness(
                witness_id="s21l:slot-mismatch",
                modifier_component_id=modifier.component_id,
                validated_anchor_role="mediator",
                validated_anchor_slot="subject",
                modifier_slot="object",
                validated_anchor_text="electric field enhancement",
                modifier_text="surface roughness",
                anchor_purity_pass=True,
                provenance_ids=["evaluation:s21l:test"],
            ),
        )


def test_higher_order_topology_id_is_deterministic():
    backbone = _backbone()

    modifier = _known(
        node_id="known:roughness-field",
        subject="surface roughness",
        relation="MODULATES",
        object_="electric field enhancement",
    )
    eligible = _eligible(modifier)

    a = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[eligible],
    )
    b = compose_higher_order_topologies(
        backbones=[backbone],
        modifiers=[eligible],
    )

    assert len(a) == 1
    assert len(b) == 1
    assert a[0].topology_id == b[0].topology_id
