from __future__ import annotations

from dac_her.hypothesis_trend_evaluation import (
    FATAL_RULE_CODES,
    NONFATAL_OBSERVATION_CODES,
    TrendHypothesisEvaluationPolicy,
    detect_claim_scope_issues,
)


def test_fatal_and_nonfatal_rule_sets_are_unique():
    assert len(FATAL_RULE_CODES) == len(set(FATAL_RULE_CODES))
    assert len(NONFATAL_OBSERVATION_CODES) == len(
        set(NONFATAL_OBSERVATION_CODES)
    )
    assert not (
        set(FATAL_RULE_CODES) & set(NONFATAL_OBSERVATION_CODES)
    )


def test_policy_has_no_hypothesis_count_acceptance_target():
    policy = TrendHypothesisEvaluationPolicy()
    assert policy.count_thresholds_used_for_acceptance is False
    assert policy.minimum_hypothesis_count is None
    assert policy.abstention_is_failure is False
    assert policy.zero_hypothesis_portfolio_is_failure is False


def test_policy_freezes_direction_frame():
    policy = TrendHypothesisEvaluationPolicy()
    assert policy.canonical_independent_change == "increase"
    assert policy.positive_direction_dependent_change == "increase"
    assert policy.negative_direction_dependent_change == "decrease"
    assert policy.sign_transformation_by_llm_allowed is False


def test_universal_overclaim_is_fatal_category():
    assert "TREND_UNIVERSAL_ESCALATION" in (
        detect_claim_scope_issues(
            "This relation always holds in all contexts.",
            cross_paper_synthesis=False,
        )
    )


def test_cross_paper_overclaim_requires_cross_paper_support():
    assert "CROSS_PAPER_OVERCLAIM" in (
        detect_claim_scope_issues(
            "This trend is replicated across papers.",
            cross_paper_synthesis=False,
        )
    )
    assert "CROSS_PAPER_OVERCLAIM" not in (
        detect_claim_scope_issues(
            "This trend is replicated across papers.",
            cross_paper_synthesis=True,
        )
    )


def test_causal_evidence_escalation_is_detected():
    assert "TREND_CAUSAL_ESCALATION" in (
        detect_claim_scope_issues(
            "The trend demonstrates a causal relationship.",
            cross_paper_synthesis=False,
        )
    )


def test_conservative_limitation_language_is_not_rejected():
    issues = detect_claim_scope_issues(
        (
            "The association is provisional rather than a causal or "
            "universal claim. Cross-paper replication is not established."
        ),
        cross_paper_synthesis=False,
    )
    assert issues == set()



def test_correct_falsifier_wording_is_not_claim_scope_overclaim():
    issues = detect_claim_scope_issues(
        (
            "Qualitative SERS performance does not increase, or instead "
            "decreases, as particle size increases under a comparable "
            "context."
        ),
        cross_paper_synthesis=False,
    )
    assert issues == set()
