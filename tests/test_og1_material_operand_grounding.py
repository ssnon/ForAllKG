from __future__ import annotations

from types import SimpleNamespace

from dac_her.og1_material_operand_grounding import (
    OG1MaterialOperandAudit,
    OG1MaterialOperandReview,
    OG1OperandSupport,
    operand_audit_passes,
    validate_operand_audit,
)


def _support(statement_id: str, excerpt: str):
    return OG1OperandSupport(
        statement_id=statement_id,
        excerpt=excerpt,
        explanation="test support",
    )


def _review(text: str, status: str, supports=(), kind="other_scientific_operand"):
    return OG1MaterialOperandReview(
        operand_text=text,
        operand_kind=kind,
        grounding_status=status,
        supports=list(supports),
        explanation="test review",
    )


def _audit(reviews, *, all_grounded=True):
    return OG1MaterialOperandAudit(
        hypothesis_id="h1",
        axis_id="axis:1",
        relation_text="local geometry is associated with formation energy",
        operand_reviews=list(reviews),
        unlisted_material_operand_texts=[],
        coverage_complete=True,
        all_material_operands_grounded=all_grounded,
        explanation="test audit",
    )


def _blueprint():
    return SimpleNamespace(
        abstain=False,
        endpoint_a=SimpleNamespace(supporting_statement_ids=["stmt:a"]),
        endpoint_b=SimpleNamespace(supporting_statement_ids=["stmt:b"]),
        novel_bridge=SimpleNamespace(
            relation="local geometry is associated with formation energy"
        ),
    )


def _context():
    return SimpleNamespace(
        evidence_statements=[
            SimpleNamespace(
                statement_id="stmt:a",
                text="The reported system contains multiple local geometry variants.",
                epistemic_role="reported",
                eligible_as_premise=True,
            ),
            SimpleNamespace(
                statement_id="stmt:b",
                text="Formation energy is reported for the matched variants.",
                epistemic_role="reported",
                eligible_as_premise=True,
            ),
        ]
    )


def _hypothesis():
    return SimpleNamespace(
        hypothesis_id="h1",
        hypothesis_statement="local geometry is associated with formation energy",
        premise_statement_ids=["stmt:a", "stmt:b"],
    )


def test_passes_when_all_operands_grounded():
    audit = _audit([
        _review("local geometry", "directly_grounded", [_support("stmt:a", "local geometry")]),
        _review("formation energy", "directly_grounded", [_support("stmt:b", "Formation energy")]),
    ])
    assert operand_audit_passes(audit)


def test_filters_unsupported_operand():
    audit = _audit([
        _review("local geometry", "directly_grounded", [_support("stmt:a", "local geometry")]),
        _review("formation energy", "unsupported"),
    ], all_grounded=False)
    assert not operand_audit_passes(audit)


def test_filters_axis_only_operand():
    audit = _audit([
        _review("local geometry", "directly_grounded", [_support("stmt:a", "local geometry")]),
        _review("formation energy", "axis_inspiration_only"),
    ], all_grounded=False)
    assert not operand_audit_passes(audit)


def test_support_excerpt_must_be_verbatim():
    audit = _audit([
        _review("local geometry", "directly_grounded", [_support("stmt:a", "not present")]),
        _review("formation energy", "directly_grounded", [_support("stmt:b", "Formation energy")]),
    ])
    issues = validate_operand_audit(
        audit,
        hypothesis=_hypothesis(),
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "nonverbatim_operand_support" in {x.code for x in issues}


def test_support_must_be_selected():
    audit = _audit([
        _review("local geometry", "directly_grounded", [_support("stmt:outside", "local geometry")]),
        _review("formation energy", "directly_grounded", [_support("stmt:b", "Formation energy")]),
    ])
    issues = validate_operand_audit(
        audit,
        hypothesis=_hypothesis(),
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "operand_support_outside_selected_premises" in {x.code for x in issues}


def test_relation_is_immutable():
    audit = _audit([
        _review("local geometry", "directly_grounded", [_support("stmt:a", "local geometry")]),
        _review("formation energy", "directly_grounded", [_support("stmt:b", "Formation energy")]),
    ]).model_copy(update={"relation_text": "changed relation"})
    issues = validate_operand_audit(
        audit,
        hypothesis=_hypothesis(),
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "relation_text_mismatch" in {x.code for x in issues}


def test_all_grounded_flag_is_recomputed():
    audit = _audit([
        _review("local geometry", "directly_grounded", [_support("stmt:a", "local geometry")]),
        _review("formation energy", "unsupported"),
    ], all_grounded=True)
    issues = validate_operand_audit(
        audit,
        hypothesis=_hypothesis(),
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "all_operands_grounded_self_audit_mismatch" in {x.code for x in issues}


def test_single_reported_premise_not_synthesis():
    audit = _audit([
        _review("local geometry", "synthesis_grounded", [_support("stmt:a", "local geometry")]),
        _review("formation energy", "directly_grounded", [_support("stmt:b", "Formation energy")]),
    ])
    issues = validate_operand_audit(
        audit,
        hypothesis=_hypothesis(),
        axis_id="axis:1",
        blueprint=_blueprint(),
        context=_context(),
    )
    assert "synthesis_without_synthesis_basis" in {x.code for x in issues}
