from pipeline_core.discovery.direct_backbone_modifier_eligibility import (
    screen_direct_backbone_candidate_modifiers,
)
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


def _known_component():
    return RelationComponentView(
        component_id="known:orientation_sers",
        subject="molecular adsorption orientation",
        relation="VARIES_WITH",
        object="relative intensities of SERS vibrational modes",
        authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        provenance=RelationComponentProvenance(
            source_kind="accepted_pattern",
            source_id="ap:orientation_sers",
            paper_id="paper:1",
            chunk_id="chunk:1",
            document_id="doc:1",
        ),
        accepted_pattern_id="ap:orientation_sers",
    )


def _backbone():
    component = _known_component()

    return DirectTaskRelationBackboneView(
        topology_id="direct_backbone:test",
        component=component,
        candidate_id="candidate:test",
        requested_source="molecular orientation",
        requested_target="Raman intensity",
        role_projection=DirectTaskRelationRoleProjection(
            requested_source="molecular orientation",
            requested_target="Raman intensity",
            source_role_slot="subject",
            target_role_slot="object",
            source_role_text=component.subject,
            target_role_text=component.object,
            projection_mode="distinct_lexical_opposite_slots",
            source_lexical_signal=LexicalRoleSignal(
                endpoint_text="molecular orientation",
                resolved=True,
                slot="subject",
                overlap_tokens=["molecular", "orientation"],
                task_coverage=1.0,
                slot_coverage=2.0 / 3.0,
            ),
            target_lexical_signal=LexicalRoleSignal(
                endpoint_text="Raman intensity",
                resolved=True,
                slot="object",
                overlap_tokens=["intensity"],
                task_coverage=0.5,
                slot_coverage=0.2,
            ),
        ),
    )


def _candidate(
    *,
    component_id: str,
    modifier: str,
):
    return RelationComponentView(
        component_id=component_id,
        subject="relative intensities of SERS vibrational modes",
        relation="VARIES_WITH",
        object=modifier,
        authority=RelationComponentAuthority.CANDIDATE_INSPIRATION,
        provenance=RelationComponentProvenance(
            source_kind="candidate_unit",
            source_id=component_id,
            source_path_id="path:test",
        ),
        candidate_unit_id=component_id,
        candidate_unit_score=0.55,
        exploration_score=0.40,
    )


def test_direct_modifier_screen_preserves_candidate_inspiration_only():
    good = _candidate(
        component_id="candidate:wavelength",
        modifier="excitation wavelength",
    )
    generic = _candidate(
        component_id="candidate:generic",
        modifier="structure",
    )

    result = screen_direct_backbone_candidate_modifiers(
        backbones=[_backbone()],
        components=[good, generic],
    )

    assert len(result.eligible_records) == 1

    row = result.eligible_records[0]

    assert row["component_id"] == "candidate:wavelength"
    assert row["anchor_role"] == "target"
    assert row["anchor_slot"] == "subject"
    assert row["modifier_slot"] == "object"
    assert row["modifier_text"] == "excitation wavelength"
    assert row["passes_frozen_screen_mirror"] is True
    assert row["reasons"] == []

    audit = result.audit

    assert audit["shadow_only"] is True
    assert audit["interaction_evidence_created"] is False
    assert audit["positive_premise_authority_created"] is False
    assert audit["novelty_authority_created"] is False
    assert audit["production_selection_authority"] is False

    generic_rows = [
        item
        for item in audit["records"]
        if item["component_id"] == "candidate:generic"
    ]

    assert len(generic_rows) == 1
    assert "GENERIC_SINGLE_TOKEN" in generic_rows[0]["reasons"]
