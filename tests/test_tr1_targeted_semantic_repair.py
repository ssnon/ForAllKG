from __future__ import annotations

from types import SimpleNamespace

from dac_her.tr1_targeted_semantic_repair import (
    TR1GroundedOperand,
    TR1GroundingSupport,
    TR1RepairPlan,
    TR1SemanticAudit,
    TR1OperandReview,
    semantic_audit_passes,
    tr1_target_eligibility,
)


def _card(
    overall="unsupported_inferential_leap",
    statuses=("unsupported_extension", "hypothesized_bridge"),
):
    reviews = [
        SimpleNamespace(status=row)
        for row in statuses
    ]
    return SimpleNamespace(
        overall_verdict=overall,
        hypothesis_clause_reviews=reviews,
        bridge_unit_reviews=[],
    )


def test_tr1_targets_localized_unsupported_inferential_leap():
    ok, _ = tr1_target_eligibility(_card())
    assert ok


def test_tr1_does_not_touch_testable_hypothesis():
    ok, _ = tr1_target_eligibility(
        _card(
            overall="testable_but_under_grounded_extension",
            statuses=("hypothesized_bridge",),
        )
    )
    assert not ok


def test_tr1_refuses_scope_mismatch_target():
    ok, _ = tr1_target_eligibility(
        _card(
            statuses=(
                "unsupported_extension",
                "scope_mismatch",
            )
        )
    )
    assert not ok


def test_tr1_repair_requires_grounded_operands_and_axis_fidelity():
    plan = TR1RepairPlan(
        source_hypothesis_id="h1",
        axis_id="axis:1",
        action="repair",
        title="repair",
        hypothesis_statement=(
            "within scope, grounded A is associated with grounded B"
        ),
        inferential_bridge=(
            "within scope, grounded A is associated with grounded B"
        ),
        premise_statement_ids=["stmt:a", "stmt:b"],
        grounded_operands=[
            TR1GroundedOperand(
                operand_text="grounded A",
                scientific_role="A",
                supports=[
                    TR1GroundingSupport(
                        statement_id="stmt:a",
                        excerpt="grounded A",
                    )
                ],
            ),
            TR1GroundedOperand(
                operand_text="grounded B",
                scientific_role="B",
                supports=[
                    TR1GroundingSupport(
                        statement_id="stmt:b",
                        excerpt="grounded B",
                    )
                ],
            ),
        ],
        removed_unsupported_material=["unsupported descriptor"],
        predicted_observable="A-B association",
        expected_direction="shift",
        prediction_rationale="tests one relation",
        falsifying_outcome="no systematic association",
        axis_fidelity_preserved=True,
        repair_reason="replace unsupported material with grounded terminals",
    )
    assert plan.action == "repair"
    assert len(plan.grounded_operands) == 2


def test_tr1_semantic_gate_passes_only_one_novel_relation_grounded_operands():
    audit = TR1SemanticAudit(
        source_hypothesis_id="h1",
        axis_id="axis:1",
        repaired_hypothesis_statement=(
            "within scope, grounded A is associated with grounded B"
        ),
        operand_reviews=[
            TR1OperandReview(
                operand_text="grounded A",
                status="directly_grounded",
                supporting_statement_ids=["stmt:a"],
                explanation="direct evidence",
            ),
            TR1OperandReview(
                operand_text="grounded B",
                status="directly_grounded",
                supporting_statement_ids=["stmt:b"],
                explanation="direct evidence",
            ),
        ],
        unlisted_material_operand_texts=[],
        relation_status="genuinely_unestablished_relation",
        one_material_relation_only=True,
        scope_compatible=True,
        axis_fidelity_preserved=True,
        explanation="valid one-edge hypothesis",
    )
    assert semantic_audit_passes(audit)


def test_tr1_semantic_gate_rejects_unsupported_operand():
    audit = TR1SemanticAudit(
        source_hypothesis_id="h1",
        axis_id="axis:1",
        repaired_hypothesis_statement="A uses new descriptor to predict B",
        operand_reviews=[
            TR1OperandReview(
                operand_text="new descriptor",
                status="unsupported",
                supporting_statement_ids=[],
                explanation="not in premises",
            )
        ],
        unlisted_material_operand_texts=[],
        relation_status="unsupported_operand",
        one_material_relation_only=True,
        scope_compatible=True,
        axis_fidelity_preserved=True,
        explanation="descriptor itself is unsupported",
    )
    assert not semantic_audit_passes(audit)


def test_tr1_semantic_gate_rejects_axis_fidelity_loss():
    audit = TR1SemanticAudit(
        source_hypothesis_id="h1",
        axis_id="axis:1",
        repaired_hypothesis_statement="A is associated with B",
        operand_reviews=[
            TR1OperandReview(
                operand_text="A",
                status="directly_grounded",
                supporting_statement_ids=["stmt:a"],
                explanation="grounded",
            ),
            TR1OperandReview(
                operand_text="B",
                status="directly_grounded",
                supporting_statement_ids=["stmt:b"],
                explanation="grounded",
            ),
        ],
        unlisted_material_operand_texts=[],
        relation_status="axis_fidelity_lost",
        one_material_relation_only=True,
        scope_compatible=True,
        axis_fidelity_preserved=False,
        explanation="repair changed scientific axis",
    )
    assert not semantic_audit_passes(audit)
