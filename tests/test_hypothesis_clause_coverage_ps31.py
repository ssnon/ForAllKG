from __future__ import annotations

from dac_her.hypothesis_clause_coverage import (
    AuditedHypothesisClause,
    ExactClauseAudit,
)
from dac_her.hypothesis_clause_coverage_v31 import (
    AuditedClauseCoverageReviewV31,
    derive_overall_verdict_v31,
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


def _review(
    local_id: str,
    status: str,
) -> AuditedClauseCoverageReviewV31:
    return AuditedClauseCoverageReviewV31(
        local_id=local_id,
        status=status,
        support_explanation="x",
        confidence="high",
    )


def test_ps31_scope_limitation_is_not_scope_conflict():
    verdict, _ = derive_overall_verdict_v31(
        clauses=[_clause("c1")],
        claim_reviews=[
            _review("c1", "evidence_scope_limitation")
        ],
        bridge_reviews=[],
    )
    assert verdict == "testable_but_under_grounded_extension"


def test_ps31_hypothesized_bridge_is_under_grounded():
    verdict, _ = derive_overall_verdict_v31(
        clauses=[_clause("c1")],
        claim_reviews=[
            _review("c1", "hypothesized_bridge")
        ],
        bridge_reviews=[],
    )
    assert verdict == "testable_but_under_grounded_extension"


def test_ps31_scope_mismatch_is_scope_conflict():
    verdict, _ = derive_overall_verdict_v31(
        clauses=[_clause("c1")],
        claim_reviews=[
            _review("c1", "scope_mismatch")
        ],
        bridge_reviews=[],
    )
    assert verdict == "scope_conflicted"


def test_ps31_real_contradiction_is_scope_conflict():
    verdict, _ = derive_overall_verdict_v31(
        clauses=[_clause("c1")],
        claim_reviews=[
            _review("c1", "contradicted_by_evidence")
        ],
        bridge_reviews=[],
    )
    assert verdict == "scope_conflicted"


def test_ps31_unsupported_extension_is_inferential_leap():
    verdict, _ = derive_overall_verdict_v31(
        clauses=[_clause("c1")],
        claim_reviews=[
            _review("c1", "unsupported_extension")
        ],
        bridge_reviews=[],
    )
    assert verdict == "unsupported_inferential_leap"


def test_ps31_scope_conflict_has_priority_over_unsupported():
    verdict, _ = derive_overall_verdict_v31(
        clauses=[
            _clause("c1"),
            _clause("c2"),
        ],
        claim_reviews=[
            _review("c1", "unsupported_extension"),
            _review("c2", "scope_mismatch"),
        ],
        bridge_reviews=[],
    )
    assert verdict == "scope_conflicted"


def test_ps31_unsupported_has_priority_over_limitation():
    verdict, _ = derive_overall_verdict_v31(
        clauses=[
            _clause("c1"),
            _clause("c2"),
        ],
        claim_reviews=[
            _review("c1", "evidence_scope_limitation"),
            _review("c2", "unsupported_extension"),
        ],
        bridge_reviews=[],
    )
    assert verdict == "unsupported_inferential_leap"


def test_ps31_all_grounded_is_grounded_extension():
    verdict, _ = derive_overall_verdict_v31(
        clauses=[
            _clause("c1"),
            _clause("c2"),
        ],
        claim_reviews=[
            _review("c1", "directly_grounded"),
            _review("c2", "synthesis_grounded"),
        ],
        bridge_reviews=[],
    )
    assert verdict == "grounded_extension"


def test_ps31_supporting_only_unsupported_does_not_downgrade_core():
    verdict, _ = derive_overall_verdict_v31(
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
