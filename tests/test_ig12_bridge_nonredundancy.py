from __future__ import annotations

from types import SimpleNamespace

from dac_her.ig11_endpoint_scope import (
    IG11AxisEvidenceAudit,
    IG11Blueprint,
    IG11BridgeScopeEnvelope,
    IG11EndpointScope,
    IG11GroundedEndpoint,
    IG11NoveltyBurdenAudit,
    IG11StatementAxisAudit,
    IG12BridgeNonRedundancyAudit,
    bridge_is_acceptable_novel_hypothesis,
    validate_axis_audit,
    validate_bridge_nonredundancy_audit,
)
from dac_her.ig1_grounded_bridge import (
    IG1DiscriminativeTest,
    IG1NovelBridge,
)


def _context():
    return SimpleNamespace(
        evidence_statements=[
            SimpleNamespace(
                statement_id="stmt:a",
                text="A reports coordination variation in supported dual sites.",
                eligible_as_premise=True,
            ),
            SimpleNamespace(
                statement_id="stmt:b",
                text="B reports structural retention in a supported model.",
                eligible_as_premise=True,
            ),
        ]
    )


def _audit():
    reviews = [
        IG11StatementAxisAudit(
            statement_id="stmt:a",
            axis_support="adjacent_context",
            endpoint_role="structural_context",
            grounding_excerpt="coordination variation",
            scope_basis_excerpt="supported dual sites",
            scope_breadth="generic_within_premise",
            scope_summary="supported sites",
            endpoint_candidate=True,
            bridge_terminal_role="bridge_terminal",
            bridge_terminal_excerpt="coordination variation",
            reason="grounded bridge terminal",
        ),
        IG11StatementAxisAudit(
            statement_id="stmt:b",
            axis_support="adjacent_context",
            endpoint_role="stability_or_outcome",
            grounding_excerpt="structural retention",
            scope_basis_excerpt="supported model",
            scope_breadth="specific_system",
            scope_summary="supported model",
            endpoint_candidate=True,
            bridge_terminal_role="bridge_terminal",
            bridge_terminal_excerpt="structural retention",
            reason="grounded bridge terminal",
        ),
    ]
    return IG11AxisEvidenceAudit(
        axis_id="axis:test",
        statement_reviews=reviews,
        axis_entities=[],
        direct_axis_statement_ids=[],
        endpoint_candidate_statement_ids=["stmt:a", "stmt:b"],
        audit_summary="two terminal endpoints",
    )


def _blueprint():
    return IG11Blueprint(
        axis_id="axis:test",
        endpoint_a=IG11GroundedEndpoint(
            endpoint_id="endpoint_a",
            anchor_statement_id="stmt:a",
            grounded_excerpt="coordination variation",
            supporting_statement_ids=["stmt:a"],
            scientific_role="coordination terminal",
            scope=IG11EndpointScope(
                scope_basis_excerpt="supported dual sites",
                scope_breadth="generic_within_premise",
                scope_summary="supported sites",
            ),
        ),
        endpoint_b=IG11GroundedEndpoint(
            endpoint_id="endpoint_b",
            anchor_statement_id="stmt:b",
            grounded_excerpt="structural retention",
            supporting_statement_ids=["stmt:b"],
            scientific_role="retention terminal",
            scope=IG11EndpointScope(
                scope_basis_excerpt="supported model",
                scope_breadth="specific_system",
                scope_summary="supported model",
            ),
        ),
        scope_envelope=IG11BridgeScopeEnvelope(
            scope_guard_phrase="within the supported model scope",
            basis_statement_ids=["stmt:a", "stmt:b"],
            system_or_material_scope="supported model scope",
            entity_or_pair_scope="reported dual sites",
            structural_or_coordination_scope="reported coordination contexts",
            observable_or_outcome_scope="retention",
        ),
        novel_bridge=IG1NovelBridge(
            subject_endpoint_id="endpoint_a",
            relation=(
                "within the supported model scope, bonding balance "
                "modulates structural retention across coordination contexts"
            ),
            object_endpoint_id="endpoint_b",
            bridge_kind="moderation",
            axis_inspiration_summary="bonding-balance axis",
        ),
        discriminative_test=IG1DiscriminativeTest(
            observable="retention versus bonding balance",
            expected_direction="shift",
            falsifying_outcome="no systematic relation",
        ),
        novelty_burden=IG11NoveltyBurdenAudit(
            direct_axis_grounding_used_when_available=True,
            burden_summary="one unestablished relation",
        ),
    )


def test_ig12_adjacent_context_can_be_bridge_terminal():
    issues = validate_axis_audit(
        _audit(),
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert issues == []


def test_ig12_genuinely_unestablished_is_acceptable():
    bp = _blueprint()
    result = IG12BridgeNonRedundancyAudit(
        axis_id="axis:test",
        bridge_relation=bp.novel_bridge.relation,
        selected_premise_statement_ids=["stmt:a", "stmt:b"],
        status="genuinely_unestablished_relation",
        directly_grounding_statement_ids=[],
        synthesis_grounding_statement_ids=[],
        scope_compatible=True,
        explanation="endpoints grounded; relation not established",
        missing_relation_if_unestablished="bonding balance -> retention",
    )
    assert validate_bridge_nonredundancy_audit(
        result,
        blueprint=bp,
        expected_axis_id="axis:test",
    ) == []
    assert bridge_is_acceptable_novel_hypothesis(result)


def test_ig12_directly_grounded_bridge_is_rejected():
    bp = _blueprint()
    result = IG12BridgeNonRedundancyAudit(
        axis_id="axis:test",
        bridge_relation=bp.novel_bridge.relation,
        selected_premise_statement_ids=["stmt:a", "stmt:b"],
        status="already_directly_grounded",
        directly_grounding_statement_ids=["stmt:a"],
        synthesis_grounding_statement_ids=[],
        scope_compatible=True,
        explanation="premise already states relation",
        missing_relation_if_unestablished=None,
    )
    assert not bridge_is_acceptable_novel_hypothesis(result)


def test_ig12_synthesis_grounded_bridge_is_rejected():
    bp = _blueprint()
    result = IG12BridgeNonRedundancyAudit(
        axis_id="axis:test",
        bridge_relation=bp.novel_bridge.relation,
        selected_premise_statement_ids=["stmt:a", "stmt:b"],
        status="already_synthesis_grounded",
        directly_grounding_statement_ids=[],
        synthesis_grounding_statement_ids=["stmt:a", "stmt:b"],
        scope_compatible=True,
        explanation="joint evidence establishes relation",
        missing_relation_if_unestablished=None,
    )
    assert not bridge_is_acceptable_novel_hypothesis(result)


def test_ig12_scope_transfer_is_rejected():
    bp = _blueprint()
    result = IG12BridgeNonRedundancyAudit(
        axis_id="axis:test",
        bridge_relation=bp.novel_bridge.relation,
        selected_premise_statement_ids=["stmt:a", "stmt:b"],
        status="scope_transfer_required",
        directly_grounding_statement_ids=[],
        synthesis_grounding_statement_ids=[],
        scope_compatible=False,
        explanation="requires unsupported transfer",
        missing_relation_if_unestablished=None,
    )
    assert not bridge_is_acceptable_novel_hypothesis(result)
