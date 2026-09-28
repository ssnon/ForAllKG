from __future__ import annotations

from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    direct_higher_order_synthesis_context,
    render_direct_higher_order_shadow_prompt,
)
from pipeline_core.discovery.direct_higher_order_topology import (
    DirectBackboneModifierEligibilityWitness,
    DirectBackboneRoleBindingView,
    DirectHigherOrderTopologyCandidate,
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


def _topology():
    known = RelationComponentView(
        component_id="known:1",
        label="orientation intensity",
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

    backbone = DirectTaskRelationBackboneView(
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

    modifier = RelationComponentView(
        component_id="candidate:1",
        label="intensity wavelength",
        subject="SERS intensity",
        relation="VARIES_WITH",
        object="excitation wavelength",
        authority=RelationComponentAuthority.CANDIDATE_INSPIRATION,
        provenance=RelationComponentProvenance(
            source_kind="candidate_unit",
            source_id="candidate_unit:1",
            source_path_id="path:1",
        ),
        candidate_unit_id="candidate_unit:1",
        candidate_unit_score=0.36,
        exploration_score=0.21,
    )

    witness = DirectBackboneModifierEligibilityWitness(
        witness_id="witness:1",
        modifier_component_id=modifier.component_id,
        validated_anchor_role="target",
        validated_anchor_slot="subject",
        modifier_slot="object",
        validated_anchor_text=modifier.subject,
        modifier_text=modifier.object,
        provenance_ids=["candidate_unit:1", "audit:s201"],
    )

    binding = DirectBackboneRoleBindingView(
        anchor_role="target",
        modifier_anchor_slot="subject",
        modifier_slot="object",
        backbone_role_text=known.subject,
        modifier_anchor_text=modifier.subject,
        modifier_text=modifier.object,
        match_mode="compatible_role_text",
        overlap_tokens=["sers", "intensity"],
        compatibility_score=1.0,
    )

    return DirectHigherOrderTopologyCandidate(
        topology_id="direct_ho:1",
        backbone_topology_id=backbone.topology_id,
        backbone=backbone,
        modifier_component=modifier,
        modifier_eligibility=witness,
        role_binding=binding,
    )


def test_context_preserves_orientation_and_authority():
    context = direct_higher_order_synthesis_context(
        topology=_topology()
    )
    direct = context.premises[0]
    modifier = context.premises[1]

    assert direct.subject == "vibrational-mode SERS intensity"
    assert direct.relation == "VARIES_WITH"
    assert (
        direct.object
        == "Raman-tensor orientation relative to the local electric field"
    )
    assert (
        direct.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
    )
    assert modifier.subject == "SERS intensity"
    assert modifier.object == "excitation wavelength"
    assert (
        modifier.authority
        == RelationComponentAuthority.CANDIDATE_INSPIRATION
    )


def test_guard_keeps_interaction_unverified():
    context = direct_higher_order_synthesis_context(
        topology=_topology()
    )
    guard = context.guard

    assert guard.topology_is_not_interaction_evidence is True
    assert guard.generated_interaction_must_remain_hypothesis is True
    assert guard.endpoint_equivalence_assertion_authorized is False
    assert guard.scientific_identity_assertion_authorized is False
    assert guard.llm_call_authorized is False


def test_rendered_prompt_marks_new_hypothesis_boundary():
    context = direct_higher_order_synthesis_context(
        topology=_topology()
    )
    prompt = render_direct_higher_order_shadow_prompt(context)

    assert "NOT evidence of an interaction" in prompt
    assert "NEW HYPOTHESIS" in prompt
    assert (
        "vibrational-mode SERS intensity --VARIES_WITH--> "
        "Raman-tensor orientation relative to the local electric field"
        in prompt
    )
    assert (
        "SERS intensity --VARIES_WITH--> excitation wavelength"
        in prompt
    )


def test_context_creates_no_novelty_or_production_authority():
    context = direct_higher_order_synthesis_context(
        topology=_topology()
    )

    assert context.shadow_only is True
    assert context.novelty_authority_created is False
    assert context.positive_premise_authority_created is False
    assert context.production_selection_authority is False
