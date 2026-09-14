from __future__ import annotations

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.relation_component_composition import (
    candidate_inspiration_component,
    compose_relation_component_topologies,
    confirmed_known_component_from_mapping,
    topology_to_task_bridge_composite,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
)
from pipeline_core.discovery.task_bridge_composite_axis import (
    materialize_task_bridge_composite_axis,
)


def _accepted_pattern() -> dict[str, object]:
    source_phrase = (
        "interparticle nanogaps promote "
        "local electromagnetic-field enhancement"
    )

    return {
        "node_id": "known:nanogap_field",
        "source_local_id": "known:nanogap_field",
        "concept_type": "RelationPattern",
        "label": source_phrase,
        "source_phrase": source_phrase,
        "description": "",
        "retention_lane": "accepted_pattern",
        "evidence_scope": "paper_result",
        "pattern_subject": "interparticle nanogaps",
        "pattern_relation": "PROMOTES",
        "pattern_object": (
            "local electromagnetic-field enhancement"
        ),
        "relation_strength": "causal_interpretive",
        "qualifiers_json": "[]",
        "pattern_support_mode": "explicit_single_span",
        "supporting_phrases_json": (
            "[\""
            + source_phrase
            + "\"]"
        ),
        "subject_evidence_phrase": "interparticle nanogaps",
        "relation_evidence_phrase": "promote",
        "object_evidence_phrase": (
            "local electromagnetic-field enhancement"
        ),
        "comparison_items_json": "[]",
        "paper_id": "paper:test",
        "chunk_id": "chunk:test",
        "document_id": "document:test",
    }


def _candidate_component():
    relation = CandidateRelationView(
        unit_id="candidate:sers_field",
        label="candidate:sers_field",
        proposed_subject="SERS intensity",
        proposed_relation="VARIES_WITH",
        proposed_object=(
            "electromagnetic field enhancement"
        ),
    )

    return candidate_inspiration_component(
        relation=relation,
        candidate_unit_score=0.60,
        exploration_score=0.20,
        quality_eligible=True,
        source_path_id="path:candidate:sers_field",
    )


def _candidate_axis() -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id="axis:candidate",
        axis_rank=1,
        inspiration_id="inspiration:candidate",
        source_path_id="path:candidate:sers_field",
        candidate_unit_id="candidate:sers_field",
        label="candidate anchor",
        entry_anchor_id="entry",
        entry_anchor_label="old source",
        exit_anchor_id="exit",
        exit_anchor_label="old target",
        proposed_subject="SERS intensity",
        proposed_relation="VARIES_WITH",
        proposed_object=(
            "electromagnetic field enhancement"
        ),
        rendered_path="candidate path",
        source_mode="exploratory",
        exploration_score=0.20,
        candidate_unit_score=0.60,
        planner_score=0.70,
        mechanistic_continuity_band="high",
        generic_entity_fraction=0.0,
        registry_hop_fraction=0.0,
        grounding_semantic_overlap=0.0,
        reaction_domain_switch_penalty=0.0,
        requires_verification=True,
        reason_codes=["candidate_unit_traversal"],
    )


def test_known_source_uses_real_target_candidate_as_axis_provenance():
    known = confirmed_known_component_from_mapping(
        _accepted_pattern()
    )
    candidate = _candidate_component()

    topology = compose_relation_component_topologies(
        components=[known, candidate],
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
    )[0]

    composite = topology_to_task_bridge_composite(
        topology
    )

    axis = materialize_task_bridge_composite_axis(
        composite=composite,
        source_axis=_candidate_axis(),
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
        axis_rank=1,
    )

    assert (
        axis.candidate_unit_id
        == "candidate:sers_field"
    )
    assert (
        axis.source_mode
        == "task_conditioned_relation_component_topology"
    )
    assert (
        axis.proposed_relation
        == "MAY_RELATE_TO_VIA_RELATION_COMPONENT_TOPOLOGY"
    )
    assert (
        "CONFIRMED KNOWN COMPONENT"
        in axis.rendered_path
    )
    assert (
        "CANDIDATE INSPIRATION COMPONENT"
        in axis.rendered_path
    )
    assert (
        "source_component_authority:confirmed_known"
        in axis.reason_codes
    )
    assert (
        "target_component_authority:candidate_inspiration"
        in axis.reason_codes
    )
    assert axis.requires_verification
