from __future__ import annotations

from dac_her.premise_role_necessity import (
    PremiseAblationReview,
    PremiseRoleReview,
    _audit_quote,
    _combine_verdict,
)


def _role(
    role: str,
    confidence: str = "high",
) -> PremiseRoleReview:
    return PremiseRoleReview(
        hypothesis_id="h",
        premise_statement_id="p",
        role=role,
        support_explanation="x",
        confidence=confidence,
    )


def _ablation(
    status: str,
    confidence: str = "high",
) -> PremiseAblationReview:
    return PremiseAblationReview(
        hypothesis_id="h",
        omitted_premise_statement_id="p",
        remaining_grounding_status=status,
        inferential_bridge_grounded=(
            status == "sufficiently_grounded"
        ),
        remaining_support_summary="x",
        confidence=confidence,
    )


def test_ps2_direct_support_plus_failed_ablation_is_critical():
    verdict, _ = _combine_verdict(
        _role("direct_clause_support"),
        _ablation("insufficiently_grounded"),
    )
    assert verdict == "critical_for_current_grounded_chain"


def test_ps2_direct_support_plus_sufficient_ablation_is_replaceable():
    verdict, _ = _combine_verdict(
        _role("direct_clause_support"),
        _ablation("sufficiently_grounded"),
    )
    assert verdict == "redundant_or_replaceable_for_current_chain"


def test_ps2_contextual_plus_sufficient_ablation_is_nonessential():
    verdict, _ = _combine_verdict(
        _role("contextual_support"),
        _ablation("sufficiently_grounded"),
    )
    assert verdict == "contextual_or_nonessential_for_current_chain"


def test_ps2_low_confidence_stays_uncertain():
    verdict, _ = _combine_verdict(
        _role("direct_clause_support", confidence="low"),
        _ablation("insufficiently_grounded"),
    )
    assert verdict == "uncertain"


def test_ps2_quote_audit_requires_exact_substring():
    valid = _audit_quote(
        "metal-metal distance",
        hypothesis_statement=(
            "Pair identity controls metal-metal distance."
        ),
        inferential_bridge="bridge",
        expected_field="hypothesis_statement",
    )
    invalid = _audit_quote(
        "metal–metal distance",
        hypothesis_statement=(
            "Pair identity controls metal-metal distance."
        ),
        inferential_bridge="bridge",
        expected_field="hypothesis_statement",
    )

    assert valid is not None
    assert valid.exact_substring_match is True
    assert invalid is not None
    assert invalid.exact_substring_match is False
