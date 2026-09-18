from __future__ import annotations

import json

from pipeline_core.discovery.higher_order_hypothesis_context import (
    materialize_higher_order_hypothesis_context,
)
from pipeline_core.discovery.higher_order_modifier_eligibility import (
    screen_confirmed_known_modifiers,
)
from pipeline_core.discovery.higher_order_synthesis_context import (
    higher_order_synthesis_context,
)
from pipeline_core.discovery.higher_order_topology_carrier import (
    topology_native_synthesis_carrier,
)
from pipeline_core.discovery.higher_order_topology_composition import (
    compose_higher_order_topologies,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.relation_component_composition import (
    candidate_inspiration_component,
    confirmed_known_component_from_mapping,
)
from pipeline_core.discovery.task_backbone_chain import (
    compose_three_component_task_backbones,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
)


def _accepted(node_id: str, subject: str, relation: str, object_: str):
    phrase = f"{subject} {relation.lower()} {object_}"
    return {
        "node_id": node_id,
        "source_local_id": node_id,
        "concept_type": "RelationPattern",
        "label": phrase,
        "source_phrase": phrase,
        "description": "",
        "retention_lane": "accepted_pattern",
        "evidence_scope": "paper_result",
        "pattern_subject": subject,
        "pattern_relation": relation,
        "pattern_object": object_,
        "relation_strength": "causal_interpretive",
        "qualifiers_json": "[]",
        "pattern_support_mode": "explicit_single_span",
        "supporting_phrases_json": json.dumps([phrase]),
        "subject_evidence_phrase": subject,
        "relation_evidence_phrase": relation.lower(),
        "object_evidence_phrase": object_,
        "comparison_items_json": "[]",
        "paper_id": "paper:test",
        "chunk_id": "chunk:test",
        "document_id": "document:test",
    }


def _known(node_id: str, subject: str, relation: str, object_: str):
    return confirmed_known_component_from_mapping(
        _accepted(node_id, subject, relation, object_)
    )


def _chain():
    source = _known(
        "known:source",
        "local-field enhancement",
        "VARIES_WITH",
        "plasmonic nanostructure size, shape, composition, and arrangement",
    )
    middle = _known(
        "known:middle",
        "electromagnetic-field enhancement",
        "VARIES_WITH",
        "nanostructure parameters",
    )
    target = _known(
        "known:target",
        "hotspot intensity and distribution",
        "VARIES_WITH",
        "nanostructure structural parameters",
    )

    rows = compose_three_component_task_backbones(
        components=[source, middle, target],
        requested_source="nanostructure shape",
        requested_target="electromagnetic hotspot location and intensity",
    )
    assert rows
    return rows[0]


def _candidate_modifier():
    return candidate_inspiration_component(
        relation=CandidateRelationView(
            unit_id="candidate:plasmonic-properties",
            proposed_subject="plasmonic properties",
            proposed_relation="VARIES_WITH",
            proposed_object=(
                "nanostructure size, shape, composition, and arrangement"
            ),
        ),
        candidate_unit_score=0.28,
        exploration_score=0.20,
        quality_eligible=True,
        source_path_id="path:candidate",
    )


def _source_context():
    return HypothesisContext(
        context_id="context:test",
        context_sha256="sha:test",
        source_packet_id="packet:test",
        source_packet_sha256="packetsha:test",
        source_report_id="report:test",
        source_report_sha256="reportsha:test",
        task_id="task:test",
        question="Which nanostructure shapes affect hotspot intensity?",
        corpus_id="corpus:test",
        domain_profile_id="sers_au_ag",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="stmt:positive",
                text="Canonical positive premise.",
                epistemic_role="reported",
                claim_kind="reported_result",
                eligible_as_premise=True,
            )
        ],
    )


def test_three_component_backbone_preserves_structured_endpoint_fidelity():
    chain = _chain()

    assert chain.source_binding.binding_authority == "structured"
    assert chain.target_binding.binding_authority == "structured"
    assert chain.source_binding_mode == "token_containment"
    assert chain.target_binding_mode == "coordinated_head_facet"
    assert chain.novelty_authority is False


def test_candidate_modifier_can_attach_without_premise_authority():
    chain = _chain()
    modifier = _candidate_modifier()

    screen = screen_confirmed_known_modifiers(
        backbones=[chain],
        components=[modifier],
    )
    assert len(screen.eligible) == 1

    rows = compose_higher_order_topologies(
        backbones=[chain],
        modifiers=screen.eligible,
    )
    assert len(rows) == 1
    assert (
        rows[0].modifier_component.candidate_unit_id
        == "candidate:plasmonic-properties"
    )
    assert rows[0].novelty_authority is False


def test_middle_relation_and_candidate_modifier_remain_restricted_nonpremise():
    chain = _chain()
    modifier = _candidate_modifier()

    screen = screen_confirmed_known_modifiers(
        backbones=[chain],
        components=[modifier],
    )
    topology = compose_higher_order_topologies(
        backbones=[chain],
        modifiers=screen.eligible,
    )[0]

    carrier = topology_native_synthesis_carrier(
        topology=topology,
        requested_source="nanostructure shape",
        requested_target="electromagnetic hotspot location and intensity",
    )
    context = higher_order_synthesis_context(carrier=carrier)

    assert [row.premise_role for row in context.premises] == [
        "source_backbone_relation",
        "middle_backbone_relation",
        "target_backbone_relation",
        "modifier_relation",
    ]

    projection = materialize_higher_order_hypothesis_context(
        source_context=_source_context(),
        higher_order_context=context,
    )

    assert [
        row.statement_id
        for row in projection.context.evidence_statements
        if row.eligible_as_premise
    ] == ["stmt:positive"]

    generated_ids = {
        row.statement_id
        for row in projection.materialization.generated_statement_lineage
    }
    generated = [
        row
        for row in projection.context.evidence_statements
        if row.statement_id in generated_ids
    ]

    assert len(generated) == 4
    assert all(not row.eligible_as_premise for row in generated)
    assert all(not row.eligible_as_gap for row in generated)

    modifier_lineage = next(
        row
        for row in projection.materialization.generated_statement_lineage
        if row.premise_role == "modifier_relation"
    )
    modifier_statement = next(
        row
        for row in generated
        if row.statement_id == modifier_lineage.statement_id
    )
    assert modifier_statement.requires_verification is True
