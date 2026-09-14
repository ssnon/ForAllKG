from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentView,
    candidate_inspiration_component,
    compose_relation_component_topologies,
    confirmed_known_component_from_mapping,
    topology_candidate_anchor_unit_id,
    topology_to_task_bridge_composite,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
)


def _accepted_pattern(
    *,
    node_id: str,
    subject: str,
    relation: str,
    object_: str,
) -> dict[str, object]:
    source_phrase = (
        f"{subject} {relation.lower()} {object_}"
    )

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
        "supporting_phrases_json": (
            "["
            + __import__("json").dumps(source_phrase)
            + "]"
        ),
        "subject_evidence_phrase": subject,
        "relation_evidence_phrase": relation.lower(),
        "object_evidence_phrase": object_,
        "comparison_items_json": "[]",
        "paper_id": "paper:test",
        "chunk_id": "chunk:test",
        "document_id": "document:test",
    }


def _candidate(
    *,
    unit_id: str,
    subject: str,
    relation: str,
    object_: str,
):
    relation_view = CandidateRelationView(
        unit_id=unit_id,
        label=unit_id,
        proposed_subject=subject,
        proposed_relation=relation,
        proposed_object=object_,
    )

    return candidate_inspiration_component(
        relation=relation_view,
        candidate_unit_score=0.60,
        exploration_score=0.20,
        quality_eligible=True,
        source_path_id=f"path:{unit_id}",
    )


def test_confirmed_known_component_has_no_fake_candidate_scores():
    component = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:nanogap_field",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="local electromagnetic-field enhancement",
        )
    )

    assert (
        component.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
    )
    assert component.accepted_pattern_id == "known:nanogap_field"
    assert component.candidate_unit_id is None
    assert component.candidate_unit_score is None
    assert component.exploration_score is None


def test_confirmed_known_cannot_be_constructed_with_candidate_scores():
    component = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:nanogap_field",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="local electromagnetic-field enhancement",
        )
    )

    payload = component.model_dump(mode="python")
    payload["candidate_unit_score"] = 1.0

    with pytest.raises(
        ValidationError,
        match="cannot carry candidate scores",
    ):
        RelationComponentView.model_validate(payload)


def test_candidate_adapter_requires_explicit_quality_gate_pass():
    relation = CandidateRelationView(
        unit_id="candidate:1",
        proposed_subject="surface morphology",
        proposed_relation="VARIES_WITH",
        proposed_object="local electromagnetic field enhancement",
    )

    with pytest.raises(
        ValueError,
        match="frozen discovery quality gate",
    ):
        candidate_inspiration_component(
            relation=relation,
            candidate_unit_score=0.60,
            exploration_score=0.20,
            quality_eligible=False,
        )


def test_known_plus_known_composes_role_aware_topology():
    source = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:nanogap_field",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="local electromagnetic-field enhancement",
        )
    )

    target = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:field_sers",
            subject="local electromagnetic-field enhancement",
            relation="PROMOTES",
            object_="SERS intensity",
        )
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
    )

    assert len(rows) == 1

    row = rows[0]

    assert row.source_binding.task_slot == "subject"
    assert row.source_binding.mediator_slot == "object"
    assert row.target_binding.task_slot == "object"
    assert row.target_binding.mediator_slot == "subject"

    assert row.shared_mediator_tokens == [
        "electromagnetic",
        "enhancement",
        "field",
        "local",
    ]

    assert (
        row.source_component.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
    )
    assert (
        row.target_component.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
    )

    assert row.epistemic_status == "inspiration_only"
    assert row.requires_verification


def test_known_plus_candidate_composes_without_promoting_authority():
    source = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:nanogap_field",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="local electromagnetic-field enhancement",
        )
    )

    target = _candidate(
        unit_id="candidate:sers_field",
        subject="SERS intensity",
        relation="VARIES_WITH",
        object_="electromagnetic field enhancement",
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
    )

    assert len(rows) == 1

    row = rows[0]

    assert (
        row.source_component.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
    )
    assert (
        row.target_component.authority
        == RelationComponentAuthority.CANDIDATE_INSPIRATION
    )
    assert row.epistemic_status == "inspiration_only"
    assert row.requires_verification


def test_one_token_pseudo_bridge_is_rejected():
    source = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:source",
            subject="surface morphology",
            relation="VARIES_WITH",
            object_="particle size",
        )
    )

    target = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:target",
            subject="pore size",
            relation="MODULATES",
            object_="SERS intensity",
        )
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="surface morphology",
        requested_target="SERS intensity",
    )

    assert rows == ()


def test_task_match_in_predicate_does_not_establish_slot_binding():
    source = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:source",
            subject="particle morphology",
            relation="SUPPRESSES",
            object_="electromagnetic field enhancement",
        )
    )

    target = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:target",
            subject="electromagnetic field enhancement",
            relation="PROMOTES",
            object_="SERS intensity",
        )
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="suppresses",
        requested_target="SERS intensity",
    )

    assert rows == ()


def test_topology_id_is_deterministic():
    source = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:nanogap_field",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="local electromagnetic-field enhancement",
        )
    )

    target = _candidate(
        unit_id="candidate:sers_field",
        subject="SERS intensity",
        relation="VARIES_WITH",
        object_="electromagnetic field enhancement",
    )

    kwargs = {
        "components": [source, target],
        "requested_source": "interparticle nanogaps",
        "requested_target": "SERS intensity",
    }

    a = compose_relation_component_topologies(**kwargs)
    b = compose_relation_component_topologies(**kwargs)

    assert len(a) == 1
    assert len(b) == 1
    assert a[0].topology_id == b[0].topology_id


def test_known_known_topology_has_no_fake_axis_anchor():
    source = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:nanogap_field",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="local electromagnetic-field enhancement",
        )
    )
    target = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:field_sers",
            subject="local electromagnetic-field enhancement",
            relation="PROMOTES",
            object_="SERS intensity",
        )
    )

    topology = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
    )[0]

    assert topology_candidate_anchor_unit_id(topology) is None

    with pytest.raises(
        ValueError,
        match="no candidate provenance anchor",
    ):
        topology_to_task_bridge_composite(topology)


def test_mixed_topology_uses_real_candidate_as_axis_anchor():
    source = confirmed_known_component_from_mapping(
        _accepted_pattern(
            node_id="known:nanogap_field",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="local electromagnetic-field enhancement",
        )
    )
    target = _candidate(
        unit_id="candidate:sers_field",
        subject="SERS intensity",
        relation="VARIES_WITH",
        object_="electromagnetic field enhancement",
    )

    topology = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
    )[0]

    composite = topology_to_task_bridge_composite(
        topology
    )

    assert (
        composite.provenance_candidate_unit_id
        == "candidate:sers_field"
    )
    assert (
        composite.source_component_authority
        == "confirmed_known"
    )
    assert (
        composite.target_component_authority
        == "candidate_inspiration"
    )
    assert composite.topology_id == topology.topology_id
    assert (
        composite.composition_mode
        == "relation_component_topology_v1"
    )


def test_require_confirmed_known_excludes_candidate_only_topology():
    source = _candidate(
        unit_id="candidate:source",
        subject="surface morphology",
        relation="VARIES_WITH",
        object_="electromagnetic field enhancement",
    )
    target = _candidate(
        unit_id="candidate:target",
        subject="SERS intensity",
        relation="VARIES_WITH",
        object_="electromagnetic field enhancement",
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="surface morphology",
        requested_target="SERS intensity",
        require_confirmed_known=True,
    )

    assert rows == ()
