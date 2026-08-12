from __future__ import annotations

from types import SimpleNamespace

from dac_her.sc1_endpoint_scope_compatibility import (
    SC1EndpointPairScopeAudit,
    SC1EndpointScopeSummary,
    SC1ScopeSupport,
    scope_audit_passes,
    validate_scope_audit,
)


def _endpoint(endpoint_id, anchor):
    return SimpleNamespace(
        endpoint_id=endpoint_id,
        anchor_statement_id=anchor,
        supporting_statement_ids=[anchor],
    )


def _blueprint():
    return SimpleNamespace(
        abstain=False,
        endpoint_a=_endpoint("endpoint_a", "stmt:a"),
        endpoint_b=_endpoint("endpoint_b", "stmt:b"),
        novel_bridge=SimpleNamespace(
            relation="within shared scope, A is associated with B"
        ),
    )


def _context():
    return SimpleNamespace(
        evidence_statements=[
            SimpleNamespace(
                statement_id="stmt:a",
                text="System X belongs to family F and reports endpoint A.",
                eligible_as_premise=True,
            ),
            SimpleNamespace(
                statement_id="stmt:b",
                text="Family F models report endpoint B across variants.",
                eligible_as_premise=True,
            ),
        ]
    )


def _audit(status, compatible, compatibility=(), transfer=(), limiting=()):
    return SC1EndpointPairScopeAudit(
        hypothesis_id="h1",
        axis_id="axis:1",
        endpoint_a_anchor_statement_id="stmt:a",
        endpoint_b_anchor_statement_id="stmt:b",
        proposed_relation="within shared scope, A is associated with B",
        endpoint_a_scope=SC1EndpointScopeSummary(
            endpoint_id="endpoint_a",
            anchor_statement_id="stmt:a",
            demonstrated_scope_summary="System X in family F",
            concrete_system_or_family_labels=["System X", "family F"],
            scope_breadth="specific_system",
        ),
        endpoint_b_scope=SC1EndpointScopeSummary(
            endpoint_id="endpoint_b",
            anchor_statement_id="stmt:b",
            demonstrated_scope_summary="family F variants",
            concrete_system_or_family_labels=["family F"],
            scope_breadth="multi_system",
        ),
        status=status,
        scope_compatible=compatible,
        compatibility_supports=list(compatibility),
        transfer_supports=list(transfer),
        limiting_supports=list(limiting),
        relation_requires_scope_pairing=True,
        comparison_basis="test basis",
        explanation="test explanation",
        missing_scope_link=(
            None if compatible else "missing transfer basis"
        ),
    )


def _support(statement_id, excerpt, kind):
    return SC1ScopeSupport(
        statement_id=statement_id,
        excerpt=excerpt,
        support_kind=kind,
        explanation="test",
    )


def test_sc1_shared_family_can_pass_with_explicit_support():
    result = _audit(
        "shared_explicit_family",
        True,
        compatibility=[
            _support(
                "stmt:a",
                "belongs to family F",
                "shared_family_basis",
            ),
            _support(
                "stmt:b",
                "Family F models",
                "shared_family_basis",
            ),
        ],
    )
    assert scope_audit_passes(result)


def test_sc1_cross_system_unjustified_fails_gate():
    result = _audit(
        "cross_system_transfer_unjustified",
        False,
        limiting=[
            _support(
                "stmt:a",
                "System X",
                "scope_limitation",
            )
        ],
    )
    assert not scope_audit_passes(result)


def test_sc1_transfer_supported_requires_transfer_support():
    result = _audit(
        "cross_system_transfer_supported",
        True,
    )
    hypothesis = SimpleNamespace(hypothesis_id="h1")
    issues = validate_scope_audit(
        result,
        hypothesis=hypothesis,
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "transfer_supported_without_transfer_support" in {
        row.code for row in issues
    }


def test_sc1_explicit_transfer_support_passes_validation():
    result = _audit(
        "cross_system_transfer_supported",
        True,
        transfer=[
            _support(
                "stmt:b",
                "across variants",
                "cross_system_transfer_basis",
            )
        ],
    )
    hypothesis = SimpleNamespace(hypothesis_id="h1")
    issues = validate_scope_audit(
        result,
        hypothesis=hypothesis,
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert issues == []


def test_sc1_scope_status_boolean_must_be_self_consistent():
    result = _audit(
        "cross_system_transfer_unjustified",
        True,
        limiting=[
            _support(
                "stmt:a",
                "System X",
                "scope_limitation",
            )
        ],
    )
    hypothesis = SimpleNamespace(hypothesis_id="h1")
    issues = validate_scope_audit(
        result,
        hypothesis=hypothesis,
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "scope_compatible_status_mismatch" in {
        row.code for row in issues
    }


def test_sc1_support_excerpt_must_be_verbatim():
    result = _audit(
        "shared_explicit_family",
        True,
        compatibility=[
            _support(
                "stmt:a",
                "not actually present",
                "shared_family_basis",
            )
        ],
    )
    hypothesis = SimpleNamespace(hypothesis_id="h1")
    issues = validate_scope_audit(
        result,
        hypothesis=hypothesis,
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "nonverbatim_scope_support" in {
        row.code for row in issues
    }


def test_sc1_relation_is_immutable():
    result = _audit(
        "shared_explicit_family",
        True,
        compatibility=[
            _support(
                "stmt:a",
                "belongs to family F",
                "shared_family_basis",
            )
        ],
    )
    result = result.model_copy(
        update={"proposed_relation": "changed relation"}
    )
    hypothesis = SimpleNamespace(hypothesis_id="h1")
    issues = validate_scope_audit(
        result,
        hypothesis=hypothesis,
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "proposed_relation_mismatch" in {
        row.code for row in issues
    }
