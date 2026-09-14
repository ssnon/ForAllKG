from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.relation_component_composition import (
    EndpointEquivalenceWitness,
    RelationComponentBindingView,
    candidate_inspiration_component,
    compose_relation_component_topologies,
    confirmed_known_component_from_mapping,
    topology_has_materializable_endpoint_fidelity,
    topology_to_task_bridge_composite,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
)


def _accepted(*, node_id: str, subject: str, relation: str, object_: str) -> dict[str, object]:
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
        "supporting_phrases_json": json.dumps([source_phrase]),
        "subject_evidence_phrase": subject,
        "relation_evidence_phrase": relation.lower(),
        "object_evidence_phrase": object_,
        "comparison_items_json": "[]",
        "paper_id": "paper:test",
        "chunk_id": "chunk:test",
        "document_id": "document:test",
    }


def _known(*, node_id: str, subject: str, relation: str, object_: str):
    return confirmed_known_component_from_mapping(
        _accepted(
            node_id=node_id,
            subject=subject,
            relation=relation,
            object_=object_,
        )
    )


def _candidate(*, unit_id: str, subject: str, relation: str, object_: str):
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


def _exact_target():
    return _known(
        node_id="known:field_sers",
        subject="electric field enhancement",
        relation="PROMOTES",
        object_="SERS intensity",
    )


def test_exact_endpoint_binding_is_materializable():
    source = _known(
        node_id="known:source",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle separation",
    )
    target = _exact_target()

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        require_endpoint_fidelity=True,
    )

    assert len(rows) == 1
    topology = rows[0]
    assert topology.source_binding.binding_authority == "exact"
    assert topology.target_binding.binding_authority == "exact"
    assert topology_has_materializable_endpoint_fidelity(topology)


def test_partial_source_binding_remains_diagnostic_but_not_materializable():
    source = _known(
        node_id="known:source",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle distance",
    )
    target = _exact_target()

    diagnostic = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )
    strict = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        require_endpoint_fidelity=True,
    )

    assert len(diagnostic) == 1
    assert diagnostic[0].source_binding.binding_authority == "partial"
    assert diagnostic[0].source_binding.task_coverage == pytest.approx(0.5)
    assert strict == ()


def test_partial_target_binding_does_not_gain_task_authority_from_intensity_only():
    source = _known(
        node_id="known:source",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle separation",
    )
    target = _known(
        node_id="known:target",
        subject="electric field enhancement",
        relation="PROMOTES",
        object_="Raman intensity",
    )

    diagnostic = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
    )
    strict = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        require_endpoint_fidelity=True,
    )

    assert len(diagnostic) == 1
    assert diagnostic[0].target_binding.binding_authority == "partial"
    assert strict == ()


def test_compound_source_slot_is_not_exact_from_one_matching_atom():
    source = _known(
        node_id="known:source",
        subject="plasmon coupling",
        relation="VARIES_WITH",
        object_="particle size, interparticle spacing, and material",
    )
    target = _known(
        node_id="known:target",
        subject="plasmon coupling",
        relation="PROMOTES",
        object_="SERS intensity",
    )

    diagnostic = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle spacing",
        requested_target="SERS intensity",
    )
    strict = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle spacing",
        requested_target="SERS intensity",
        require_endpoint_fidelity=True,
    )

    assert len(diagnostic) == 1
    assert diagnostic[0].source_binding.binding_authority == "partial"
    assert strict == ()


def test_hyphenation_is_orthographic_not_semantic_equivalence():
    source = _known(
        node_id="known:source",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="inter-particle distance",
    )
    target = _exact_target()

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle distance",
        requested_target="SERS intensity",
        require_endpoint_fidelity=True,
    )

    assert len(rows) == 1
    assert rows[0].source_binding.binding_authority == "exact"
    assert rows[0].source_binding.equivalence_witness_id is None


def test_explicit_equivalence_witness_can_promote_binding_authority():
    source = _known(
        node_id="known:source",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle distance",
    )
    target = _exact_target()

    witness = EndpointEquivalenceWitness(
        witness_id="eq:distance-separation",
        left_endpoint="interparticle separation",
        right_endpoint="interparticle distance",
        witness_kind="task_supplied",
        provenance_ids=["review:s22c:test"],
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        endpoint_equivalences=[witness],
        require_endpoint_fidelity=True,
    )

    assert len(rows) == 1
    topology = rows[0]
    assert topology.source_binding.binding_authority == "equivalent"
    assert topology.source_binding.equivalence_witness_id == "eq:distance-separation"
    assert (
        "source_endpoint_equivalence_witness:eq:distance-separation"
        in topology.reason_codes
    )


def test_equivalent_binding_without_witness_fails_closed():
    with pytest.raises(
        ValidationError,
        match="requires an explicit witness",
    ):
        RelationComponentBindingView(
            task_slot="object",
            mediator_slot="subject",
            binding_authority="equivalent",
            matched_endpoint_atom="interparticle separation",
        )


def test_partial_mixed_topology_cannot_bypass_direct_materializer():
    source = _known(
        node_id="known:source",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle distance",
    )
    target = _candidate(
        unit_id="candidate:target",
        subject="electric field enhancement",
        relation="PROMOTES",
        object_="SERS intensity",
    )

    topology = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        require_candidate_anchor=True,
    )[0]

    assert not topology_has_materializable_endpoint_fidelity(topology)

    with pytest.raises(
        ValueError,
        match="partial endpoint binding",
    ):
        topology_to_task_bridge_composite(topology)


def test_mediator_compatibility_score_is_normalized():
    source = _known(
        node_id="known:source",
        subject="electric field intensity",
        relation="VARIES_WITH",
        object_="interparticle separation",
    )
    target = _known(
        node_id="known:target",
        subject="electric field enhancement",
        relation="PROMOTES",
        object_="SERS intensity",
    )

    rows = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        require_endpoint_fidelity=True,
    )

    assert len(rows) == 1
    assert 0.0 <= rows[0].compatibility_score <= 1.0
    assert rows[0].compatibility_score < 1.0


def test_exact_mediator_identity_is_maximum_score():
    source = _known(
        node_id="known:source",
        subject="electric field enhancement",
        relation="VARIES_WITH",
        object_="interparticle separation",
    )
    target = _exact_target()

    row = compose_relation_component_topologies(
        components=[source, target],
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        require_endpoint_fidelity=True,
    )[0]

    assert row.compatibility_score == pytest.approx(1.0)
