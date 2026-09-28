from __future__ import annotations

from pipeline_core.discovery.direct_higher_order_topology import compose_direct_higher_order_topologies
from pipeline_core.discovery.direct_task_relation_backbone import (
    DirectTaskRelationBackboneView,
    DirectTaskRelationRoleProjection,
    LexicalRoleSignal,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentProvenance,
    RelationComponentView,
)


def _backbone():
    known = RelationComponentView(
        component_id="known:1",
        label="orientation-intensity",
        subject="vibrational-mode SERS intensity",
        relation="VARIES_WITH",
        object="Raman-tensor orientation relative to the local electric field",
        authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        provenance=RelationComponentProvenance(
            source_kind="accepted_pattern",
            source_id="accepted:1",
            paper_id="paper:1",
            chunk_id="chunk:1",
            document_id="doc:1",
            evidence_scope="paper_result",
            pattern_support_mode="explicit_single_span",
            relation_strength="correlational",
        ),
        accepted_pattern_id="accepted:1",
    )
    return DirectTaskRelationBackboneView(
        topology_id="direct:1",
        component=known,
        candidate_id="rp:1",
        requested_source="molecular orientation",
        requested_target="Raman intensity",
        role_projection=DirectTaskRelationRoleProjection(
            requested_source="molecular orientation",
            requested_target="Raman intensity",
            source_role_slot="object",
            target_role_slot="subject",
            source_role_text=known.object,
            target_role_text=known.subject,
            projection_mode="distinct_lexical_opposite_slots",
            source_lexical_signal=LexicalRoleSignal(
                endpoint_text="molecular orientation",
                resolved=True,
                slot="object",
                overlap_tokens=["orientation"],
                task_coverage=0.5,
                slot_coverage=0.2,
            ),
            target_lexical_signal=LexicalRoleSignal(
                endpoint_text="Raman intensity",
                resolved=True,
                slot="subject",
                overlap_tokens=["intensity"],
                task_coverage=0.5,
                slot_coverage=0.33,
            ),
        ),
    )


def _modifier(subject="SERS intensity", object_="Ag capping thickness"):
    return RelationComponentView(
        component_id="candidate:1",
        label="modifier",
        subject=subject,
        relation="VARIES_WITH",
        object=object_,
        authority=RelationComponentAuthority.CANDIDATE_INSPIRATION,
        provenance=RelationComponentProvenance(
            source_kind="candidate_unit",
            source_id="candidate_unit:1",
            source_path_id="path:1",
        ),
        candidate_unit_id="candidate_unit:1",
        candidate_unit_score=0.35,
        exploration_score=0.31,
    )


def _record(component):
    return {
        "component_id": component.component_id,
        "subject": component.subject,
        "relation": component.relation,
        "object": component.object,
        "anchor_role": "target",
        "anchor_slot": "subject",
        "modifier_slot": "object",
        "anchor_text": component.subject,
        "modifier_text": component.object,
        "reasons": [],
        "passes_frozen_screen_mirror": True,
    }


def test_target_role_modifier_materializes():
    b = _backbone()
    m = _modifier()
    rows = compose_direct_higher_order_topologies(
        backbones=[b],
        modifier_components=[m],
        eligible_modifier_records=[_record(m)],
        audit_source="audit:s201",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.role_binding.anchor_role == "target"
    assert row.role_binding.match_mode == "compatible_role_text"
    assert row.role_binding.compatibility_score == 1.0
    assert row.role_binding.modifier_text == "Ag capping thickness"


def test_nonmatching_role_does_not_materialize():
    b = _backbone()
    m = _modifier(subject="surface chemistry")
    rows = compose_direct_higher_order_topologies(
        backbones=[b],
        modifier_components=[m],
        eligible_modifier_records=[_record(m)],
        audit_source="audit:s201",
    )
    assert rows == ()


def test_failed_upstream_screen_fails_closed():
    b = _backbone()
    m = _modifier()
    record = _record(m)
    record["passes_frozen_screen_mirror"] = False
    try:
        compose_direct_higher_order_topologies(
            backbones=[b],
            modifier_components=[m],
            eligible_modifier_records=[record],
            audit_source="audit:s201",
        )
    except ValueError as exc:
        assert "did not pass frozen screen mirror" in str(exc)
    else:
        raise AssertionError("expected fail-closed rejection")


def test_topology_has_no_interaction_or_novelty_authority():
    b = _backbone()
    m = _modifier()
    row = compose_direct_higher_order_topologies(
        backbones=[b],
        modifier_components=[m],
        eligible_modifier_records=[_record(m)],
        audit_source="audit:s201",
    )[0]
    assert row.shadow_only is True
    assert row.topology_is_interaction_evidence is False
    assert row.interaction_claim_authorized is False
    assert row.novelty_authority_created is False
    assert row.positive_premise_authority_created is False
    assert row.production_selection_authority is False
