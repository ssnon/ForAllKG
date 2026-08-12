from __future__ import annotations

from types import SimpleNamespace

from dac_her.ig11_endpoint_scope import (
    IG11AxisEntityAudit,
    IG11AxisEvidenceAudit,
    IG11Blueprint,
    IG11BridgeScopeEnvelope,
    IG11EndpointScope,
    IG11GroundedEndpoint,
    IG11NoveltyBurdenAudit,
    IG11StatementAxisAudit,
    validate_axis_audit,
    validate_blueprint,
)
from dac_her.ig1_grounded_bridge import IG1DiscriminativeTest, IG1NovelBridge


def _context():
    return SimpleNamespace(evidence_statements=[
        SimpleNamespace(
            statement_id="stmt:a",
            text="The supplied dual-atom models report hydrogen adsorption energetics and catalytic activity.",
            eligible_as_premise=True,
        ),
        SimpleNamespace(
            statement_id="stmt:b",
            text="The supplied study reports structural retention for a supported dual-atom model.",
            eligible_as_premise=True,
        ),
    ])


def _audit(entity):
    reviews=[
        IG11StatementAxisAudit(
            statement_id="stmt:a",
            axis_support="direct_axis_grounding",
            endpoint_role="reaction_or_activity_outcome",
            grounding_excerpt="hydrogen adsorption energetics",
            scope_basis_excerpt="supplied dual-atom models",
            scope_breadth="multi_system",
            scope_summary="supplied dual-atom models",
            endpoint_candidate=True,
            bridge_terminal_role="both",
            bridge_terminal_excerpt="hydrogen adsorption energetics",
            reason="adsorption terminal",
        ),
        IG11StatementAxisAudit(
            statement_id="stmt:b",
            axis_support="adjacent_context",
            endpoint_role="stability_or_outcome",
            grounding_excerpt="structural retention",
            scope_basis_excerpt="supported dual-atom model",
            scope_breadth="specific_system",
            scope_summary="supported model",
            endpoint_candidate=True,
            bridge_terminal_role="bridge_terminal",
            bridge_terminal_excerpt="structural retention",
            reason="retention terminal",
        ),
    ]
    return IG11AxisEvidenceAudit(
        axis_id="axis:test",
        statement_reviews=reviews,
        axis_entities=[entity],
        direct_axis_statement_ids=["stmt:a"],
        endpoint_candidate_statement_ids=["stmt:a","stmt:b"],
        audit_summary="taxonomy test",
    )


def _blueprint(relation):
    return IG11Blueprint(
        axis_id="axis:test",
        endpoint_a=IG11GroundedEndpoint(
            endpoint_id="endpoint_a", anchor_statement_id="stmt:a",
            grounded_excerpt="hydrogen adsorption energetics",
            supporting_statement_ids=["stmt:a"], scientific_role="adsorption terminal",
            scope=IG11EndpointScope(
                scope_basis_excerpt="supplied dual-atom models",
                scope_breadth="multi_system", scope_summary="supplied models")),
        endpoint_b=IG11GroundedEndpoint(
            endpoint_id="endpoint_b", anchor_statement_id="stmt:b",
            grounded_excerpt="structural retention",
            supporting_statement_ids=["stmt:b"], scientific_role="retention terminal",
            scope=IG11EndpointScope(
                scope_basis_excerpt="supported dual-atom model",
                scope_breadth="specific_system", scope_summary="supported model")),
        scope_envelope=IG11BridgeScopeEnvelope(
            scope_guard_phrase="within the supplied supported models",
            basis_statement_ids=["stmt:a","stmt:b"],
            system_or_material_scope="supplied supported models",
            entity_or_pair_scope="reported dual-atom sites",
            structural_or_coordination_scope="reported structures",
            observable_or_outcome_scope="adsorption/desorption and retention"),
        novel_bridge=IG1NovelBridge(
            subject_endpoint_id="endpoint_a", relation=relation,
            object_endpoint_id="endpoint_b", bridge_kind="descriptor_link",
            axis_inspiration_summary="adsorption-desorption compatibility"),
        discriminative_test=IG1DiscriminativeTest(
            observable="retention versus adsorption/desorption descriptor",
            expected_direction="shift", falsifying_outcome="no systematic dependence"),
        novelty_burden=IG11NoveltyBurdenAudit(
            direct_axis_grounding_used_when_available=True,
            burden_summary="one relation"),
    )


def test_ig12a_reaction_species_entity_kind_is_valid():
    entity=IG11AxisEntityAudit(
        entity_text="H2", entity_kind="reaction_species_or_product",
        grounding_status="ungrounded", grounding_statement_ids=[],
        entity_specific_claim_required_for_axis_fidelity=False,
        reason="reaction product, not catalyst identity")
    issues=validate_axis_audit(_audit(entity), context=_context(), expected_axis_id="axis:test")
    assert issues == []


def test_ig12a_reaction_species_does_not_trigger_concrete_identity_gate():
    entity=IG11AxisEntityAudit(
        entity_text="H2", entity_kind="reaction_species_or_product",
        grounding_status="ungrounded", grounding_statement_ids=[],
        entity_specific_claim_required_for_axis_fidelity=False,
        reason="reaction product")
    issues=validate_blueprint(
        _blueprint("within the supplied supported models, H2-related desorption compatibility may track structural retention"),
        audit=_audit(entity), context=_context(), expected_axis_id="axis:test")
    codes={x.code for x in issues}
    assert "required_axis_entity_ungrounded" not in codes
    assert "ungrounded_concrete_axis_entity_in_relation" not in codes


def test_ig12a_concrete_metal_pair_still_triggers_gate():
    entity=IG11AxisEntityAudit(
        entity_text="Co/Mo", entity_kind="concrete_material_or_system",
        grounding_status="ungrounded", grounding_statement_ids=[],
        entity_specific_claim_required_for_axis_fidelity=True,
        reason="specific metal pair defines axis")
    issues=validate_blueprint(
        _blueprint("within the supplied supported models, Co/Mo identity may modulate structural retention"),
        audit=_audit(entity), context=_context(), expected_axis_id="axis:test")
    codes={x.code for x in issues}
    assert "required_axis_entity_ungrounded" in codes
    assert "ungrounded_concrete_axis_entity_in_relation" in codes


def test_ig12a_species_relation_still_reaches_normal_scientific_gates():
    entity=IG11AxisEntityAudit(
        entity_text="H2", entity_kind="reaction_species_or_product",
        grounding_status="ungrounded", grounding_statement_ids=[],
        entity_specific_claim_required_for_axis_fidelity=False,
        reason="reaction product")
    issues=validate_blueprint(
        _blueprint("within the supplied supported models, H2 desorption may mediate structural retention"),
        audit=_audit(entity), context=_context(), expected_axis_id="axis:test")
    assert not any("concrete_axis_entity" in x.code or "required_axis_entity" in x.code for x in issues)
