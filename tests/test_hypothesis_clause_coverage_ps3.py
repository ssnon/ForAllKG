from __future__ import annotations

from dac_her.hypothesis_clause_coverage import (
    AuditedBridgeUnit,
    AuditedClauseCoverageReview,
    AuditedHypothesisClause,
    ExactClauseAudit,
    derive_overall_verdict,
)


def _clause(
    local_id: str,
    *,
    materiality: str = "core",
) -> AuditedHypothesisClause:
    return AuditedHypothesisClause(
        local_id=local_id,
        text=local_id,
        clause_type="central_relation",
        materiality=materiality,
        quote_audit=ExactClauseAudit(
            local_id=local_id,
            text=local_id,
            source_field="hypothesis_statement",
            exact_substring_match=True,
        ),
    )


def _bridge(local_id: str) -> AuditedBridgeUnit:
    return AuditedBridgeUnit(
        local_id=local_id,
        text=local_id,
        materiality="core",
        quote_audit=ExactClauseAudit(
            local_id=local_id,
            text=local_id,
            source_field="inferential_bridge",
            exact_substring_match=True,
        ),
    )


def _review(
    local_id: str,
    status: str,
) -> AuditedClauseCoverageReview:
    return AuditedClauseCoverageReview(
        local_id=local_id,
        status=status,
        support_explanation="x",
        confidence="high",
    )


def test_ps3_all_grounded_is_grounded_extension():
    verdict, _ = derive_overall_verdict(
        clauses=[_clause("c1")],
        claim_reviews=[_review("c1", "directly_grounded")],
        bridge_reviews=[],
    )
    assert verdict == "grounded_extension"


def test_ps3_hypothesized_bridge_is_under_grounded_extension():
    verdict, _ = derive_overall_verdict(
        clauses=[_clause("c1")],
        claim_reviews=[_review("c1", "hypothesized_bridge")],
        bridge_reviews=[],
    )
    assert verdict == "testable_but_under_grounded_extension"


def test_ps3_unsupported_core_is_inferential_leap():
    verdict, _ = derive_overall_verdict(
        clauses=[_clause("c1")],
        claim_reviews=[_review("c1", "unsupported_extension")],
        bridge_reviews=[],
    )
    assert verdict == "unsupported_inferential_leap"


def test_ps3_bridge_unsupported_is_inferential_leap():
    verdict, _ = derive_overall_verdict(
        clauses=[_clause("c1")],
        claim_reviews=[_review("c1", "directly_grounded")],
        bridge_reviews=[
            _review("b1", "unsupported_extension")
        ],
    )
    assert verdict == "unsupported_inferential_leap"


def test_ps3_limitation_has_highest_priority():
    verdict, _ = derive_overall_verdict(
        clauses=[_clause("c1")],
        claim_reviews=[
            _review("c1", "contradicted_or_limited")
        ],
        bridge_reviews=[
            _review("b1", "unsupported_extension")
        ],
    )
    assert verdict == "scope_conflicted"


def test_ps3_supporting_clause_does_not_override_core_grounding():
    verdict, _ = derive_overall_verdict(
        clauses=[
            _clause("c1", materiality="core"),
            _clause("c2", materiality="supporting"),
        ],
        claim_reviews=[
            _review("c1", "directly_grounded"),
            _review("c2", "unsupported_extension"),
        ],
        bridge_reviews=[],
    )
    assert verdict == "grounded_extension"
