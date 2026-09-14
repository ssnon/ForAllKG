from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxis
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentView,
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


def _accepted(
    *,
    node_id: str,
    subject: str,
    relation: str,
    object_: str,
) -> dict[str, object]:
    source_phrase = f"{subject} {relation.lower()} {object_}"
    return {
        "node_id": node_id,
        "source_local_id": node_id + ":local",
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
        "supporting_phrases_json": __import__("json").dumps([source_phrase]),
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
    return candidate_inspiration_component(
        relation=CandidateRelationView(
            unit_id=unit_id,
            label=unit_id,
            proposed_subject=subject,
            proposed_relation=relation,
            proposed_object=object_,
        ),
        candidate_unit_score=0.60,
        exploration_score=0.20,
        quality_eligible=True,
        source_path_id="path:" + unit_id,
    )


def _candidate_axis(unit_id: str) -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id="axis:" + unit_id,
        axis_rank=1,
        inspiration_id="inspiration:" + unit_id,
        source_path_id="path:" + unit_id,
        candidate_unit_id=unit_id,
        label="candidate anchor",
        entry_anchor_id="entry",
        entry_anchor_label="old source",
        exit_anchor_id="exit",
        exit_anchor_label="old target",
        proposed_subject="SERS intensity",
        proposed_relation="VARIES_WITH",
        proposed_object="electromagnetic field enhancement",
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


def test_confirmed_known_provenance_identity_is_fail_closed():
    component = confirmed_known_component_from_mapping(
        _accepted(
            node_id="known:a",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="electromagnetic field enhancement",
        )
    )
    payload = component.model_dump(mode="python")
    payload["provenance"]["source_id"] = "known:wrong"

    with pytest.raises(
        ValidationError,
        match="source_id must match",
    ):
        RelationComponentView.model_validate(payload)


def test_confirmed_known_requires_document_provenance():
    row = _accepted(
        node_id="known:a",
        subject="interparticle nanogaps",
        relation="PROMOTES",
        object_="electromagnetic field enhancement",
    )
    row["paper_id"] = ""

    with pytest.raises(
        ValidationError,
        match="paper/chunk/document provenance",
    ):
        confirmed_known_component_from_mapping(row)


def test_candidate_component_requires_source_path_provenance():
    relation = CandidateRelationView(
        unit_id="candidate:a",
        proposed_subject="SERS intensity",
        proposed_relation="VARIES_WITH",
        proposed_object="electromagnetic field enhancement",
    )

    with pytest.raises(
        ValidationError,
        match="source_path_id provenance",
    ):
        candidate_inspiration_component(
            relation=relation,
            candidate_unit_score=0.60,
            exploration_score=0.20,
            quality_eligible=True,
            source_path_id="",
        )


def test_materializable_topology_filter_excludes_known_known_but_keeps_mixed():
    source = confirmed_known_component_from_mapping(
        _accepted(
            node_id="known:source",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="electromagnetic field enhancement",
        )
    )
    known_target = confirmed_known_component_from_mapping(
        _accepted(
            node_id="known:target",
            subject="electromagnetic field enhancement",
            relation="PROMOTES",
            object_="SERS intensity",
        )
    )
    candidate_target = _candidate(
        unit_id="candidate:target",
        subject="SERS intensity",
        relation="VARIES_WITH",
        object_="electromagnetic field enhancement",
    )

    rows = compose_relation_component_topologies(
        components=[source, known_target, candidate_target],
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
        max_topologies=12,
        require_confirmed_known=True,
        require_candidate_anchor=True,
    )

    assert rows
    assert all(
        (
            row.source_component.authority
            == RelationComponentAuthority.CANDIDATE_INSPIRATION
        )
        or
        (
            row.target_component.authority
            == RelationComponentAuthority.CANDIDATE_INSPIRATION
        )
        for row in rows
    )
    assert any(
        row.target_component.candidate_unit_id == "candidate:target"
        for row in rows
    )


def test_axis_preserves_known_component_provenance_reason_codes():
    known = confirmed_known_component_from_mapping(
        _accepted(
            node_id="known:nanogap_field",
            subject="interparticle nanogaps",
            relation="PROMOTES",
            object_="local electromagnetic-field enhancement",
        )
    )
    candidate = _candidate(
        unit_id="candidate:sers_field",
        subject="SERS intensity",
        relation="VARIES_WITH",
        object_="electromagnetic field enhancement",
    )
    topology = compose_relation_component_topologies(
        components=[known, candidate],
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
        require_confirmed_known=True,
        require_candidate_anchor=True,
    )[0]
    composite = topology_to_task_bridge_composite(topology)
    axis = materialize_task_bridge_composite_axis(
        composite=composite,
        source_axis=_candidate_axis("candidate:sers_field"),
        requested_source="interparticle nanogaps",
        requested_target="SERS intensity",
        axis_rank=1,
    )

    assert "source_component_source_id:known:nanogap_field" in axis.reason_codes
    assert "source_component_paper_id:paper:test" in axis.reason_codes
    assert "source_component_chunk_id:chunk:test" in axis.reason_codes
    assert "source_component_document_id:document:test" in axis.reason_codes


def test_e2e_forwards_accepted_patterns_only_to_task_axis_builder():
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "discovery"
        / "run_dac_discovery_e2e.py"
    ).read_text(encoding="utf-8")

    task_pos = source.index(
        '"scripts.discovery.build_task_conditioned_axis_plan"'
    )
    task_segment = source[task_pos:task_pos + 2500]
    assert "--accepted-patterns" in task_segment

    refinement_pos = source.index(
        '"scripts.discovery.run_novelty_refinement"'
    )
    refinement_segment = source[
        refinement_pos:refinement_pos + 3500
    ]
    assert "--accepted-patterns" not in refinement_segment
