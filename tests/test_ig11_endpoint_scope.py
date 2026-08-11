from __future__ import annotations

from types import SimpleNamespace

from dac_her.ig11_endpoint_scope import (
    IG11AxisEntityAudit,
    IG11AxisEvidenceAudit,
    IG11BridgeScopeEnvelope,
    IG11Blueprint,
    IG11EndpointScope,
    IG11GroundedEndpoint,
    IG11NoveltyBurdenAudit,
    IG11StatementAxisAudit,
    validate_blueprint,
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
                text="A grounds hydrogen adsorption energetics in system X.",
                eligible_as_premise=True,
            ),
            SimpleNamespace(
                statement_id="stmt:b",
                text="B reports structural stability for supported dual sites.",
                eligible_as_premise=True,
            ),
            SimpleNamespace(
                statement_id="stmt:c",
                text="C is adjacent context only.",
                eligible_as_premise=True,
            ),
        ]
    )


def _audit(
    *,
    direct=("stmt:a",),
    entity_ungrounded=False,
):
    return IG11AxisEvidenceAudit(
        axis_id="axis:test",
        statement_reviews=[
            IG11StatementAxisAudit(
                statement_id="stmt:a",
                axis_support="direct_axis_grounding",
                endpoint_role="reaction_or_activity_outcome",
                grounding_excerpt="hydrogen adsorption energetics",
                scope_basis_excerpt="system X",
                scope_breadth="specific_system",
                scope_summary="system X",
                endpoint_candidate=True,
                reason="direct axis variable",
            ),
            IG11StatementAxisAudit(
                statement_id="stmt:b",
                axis_support="adjacent_context",
                endpoint_role="stability_or_outcome",
                grounding_excerpt="structural stability",
                scope_basis_excerpt="supported dual sites",
                scope_breadth="generic_within_premise",
                scope_summary="supported dual sites",
                endpoint_candidate=True,
                reason="stability endpoint",
            ),
            IG11StatementAxisAudit(
                statement_id="stmt:c",
                axis_support="adjacent_context",
                endpoint_role="structural_context",
                grounding_excerpt="adjacent context",
                scope_basis_excerpt="adjacent context",
                scope_breadth="unclear",
                scope_summary="adjacent",
                endpoint_candidate=True,
                reason="context",
            ),
        ],
        axis_entities=(
            [
                IG11AxisEntityAudit(
                    entity_text="Co/Mo",
                    entity_kind="concrete_material_or_system",
                    grounding_status="ungrounded",
                    grounding_statement_ids=[],
                    entity_specific_claim_required_for_axis_fidelity=True,
                    reason="specific pair defines axis",
                )
            ]
            if entity_ungrounded
            else []
        ),
        direct_axis_statement_ids=list(direct),
        endpoint_candidate_statement_ids=[
            "stmt:a",
            "stmt:b",
            "stmt:c",
        ],
        audit_summary="test",
    )


def _blueprint(
    *,
    endpoint_a="stmt:a",
    relation=(
        "Within system X, hydrogen adsorption is associated with stability."
    ),
):
    text_by_id = {
        "stmt:a": "hydrogen adsorption energetics",
        "stmt:b": "structural stability",
        "stmt:c": "adjacent context",
    }
    scope_by_id = {
        "stmt:a": "system X",
        "stmt:b": "supported dual sites",
        "stmt:c": "adjacent context",
    }
    return IG11Blueprint(
        axis_id="axis:test",
        endpoint_a=IG11GroundedEndpoint(
            endpoint_id="endpoint_a",
            anchor_statement_id=endpoint_a,
            grounded_excerpt=text_by_id[endpoint_a],
            supporting_statement_ids=[endpoint_a],
            scientific_role="endpoint A",
            scope=IG11EndpointScope(
                scope_basis_excerpt=scope_by_id[endpoint_a],
                scope_breadth="specific_system"
                if endpoint_a == "stmt:a"
                else "unclear",
                scope_summary="scope A",
            ),
        ),
        endpoint_b=IG11GroundedEndpoint(
            endpoint_id="endpoint_b",
            anchor_statement_id="stmt:b",
            grounded_excerpt="structural stability",
            supporting_statement_ids=["stmt:b"],
            scientific_role="endpoint B",
            scope=IG11EndpointScope(
                scope_basis_excerpt="supported dual sites",
                scope_breadth="generic_within_premise",
                scope_summary="scope B",
            ),
        ),
        scope_envelope=IG11BridgeScopeEnvelope(
            scope_guard_phrase="Within system X",
            basis_statement_ids=[endpoint_a, "stmt:b"],
            system_or_material_scope="system X",
            entity_or_pair_scope="selected supported sites",
            structural_or_coordination_scope="selected structures",
            observable_or_outcome_scope="adsorption and stability",
        ),
        novel_bridge=IG1NovelBridge(
            subject_endpoint_id="endpoint_a",
            relation=relation,
            object_endpoint_id="endpoint_b",
            bridge_kind="moderation",
            axis_inspiration_summary="axis",
        ),
        discriminative_test=IG1DiscriminativeTest(
            observable="adsorption-stability association",
            expected_direction="shift",
            falsifying_outcome="no association",
        ),
        novelty_burden=IG11NoveltyBurdenAudit(
            direct_axis_grounding_used_when_available=(
                endpoint_a == "stmt:a"
            ),
            burden_summary="one relation",
        ),
    )


def test_ig11_valid_blueprint_uses_direct_axis_evidence():
    issues = validate_blueprint(
        _blueprint(),
        audit=_audit(),
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert issues == []


def test_ig11_omitting_direct_axis_evidence_fails():
    bp = _blueprint(
        endpoint_a="stmt:c",
        relation=(
            "Within system X, adjacent context is associated with stability."
        ),
    )
    issues = validate_blueprint(
        bp,
        audit=_audit(),
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert "direct_axis_evidence_omitted" in {
        row.code for row in issues
    }


def test_ig11_required_ungrounded_entity_forces_active_plan_failure():
    bp = _blueprint(
        relation=(
            "Within system X, Co/Mo identity moderates stability."
        )
    )
    issues = validate_blueprint(
        bp,
        audit=_audit(entity_ungrounded=True),
        context=_context(),
        expected_axis_id="axis:test",
    )
    codes = {row.code for row in issues}
    assert "required_axis_entity_ungrounded" in codes
    assert "ungrounded_concrete_axis_entity_in_relation" in codes


def test_ig11_scope_guard_must_be_in_relation():
    try:
        _blueprint(
            relation="Hydrogen adsorption is associated with stability."
        )
    except ValueError as exc:
        assert "scope_guard_phrase" in str(exc)
    else:
        raise AssertionError("missing scope guard should fail model validation")


def test_ig11_scope_basis_cannot_use_unselected_premise():
    bp = _blueprint()
    bp.scope_envelope.basis_statement_ids.append("stmt:c")
    issues = validate_blueprint(
        bp,
        audit=_audit(),
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert "scope_basis_outside_endpoint_support" in {
        row.code for row in issues
    }


def test_ig11_multi_hop_marker_fails():
    bp = _blueprint(
        relation=(
            "Within system X, adsorption changes and then stability changes."
        )
    )
    issues = validate_blueprint(
        bp,
        audit=_audit(),
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert "multi_hop_relation_marker" in {
        row.code for row in issues
    }


def test_ig11_abstention_is_allowed():
    bp = IG11Blueprint(
        axis_id="axis:test",
        abstain=True,
        abstention_reason="required axis entity is ungrounded",
    )
    issues = validate_blueprint(
        bp,
        audit=_audit(entity_ungrounded=True),
        context=_context(),
        expected_axis_id="axis:test",
    )
    assert issues == []
