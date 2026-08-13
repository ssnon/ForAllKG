from __future__ import annotations

import json
from pathlib import Path

from dac_her.prior_art_review_audit import (
    current_prior_art_review_audit_context,
    prior_art_review_audit_scope,
    summarize_prior_art_review_audit,
)


def _row(
    request: str,
    response: str,
    *,
    kind: str,
    hypothesis: str,
    focal: str | None = None,
    cost: float = 1.0,
) -> dict:
    return {
        "record_type": "prior_art_review_call",
        "request_fingerprint": request,
        "response_fingerprint": response,
        "hypothesis_id": hypothesis,
        "provider_input_tokens": 10,
        "provider_output_tokens": 5,
        "provider_total_tokens": 15,
        "provider_cost_credits": cost,
        "assessment_context": {
            "assessment_kind": kind,
            **(
                {"focal_hypothesis_id": focal}
                if focal is not None
                else {}
            ),
        },
    }


def test_audit_scope_is_diagnostic_and_restores_context():
    assert current_prior_art_review_audit_context() == {}
    with prior_art_review_audit_scope(
        assessment_kind="alpha5_initial",
        source_portfolio_id="portfolio:A",
    ):
        assert current_prior_art_review_audit_context() == {
            "assessment_kind": "alpha5_initial",
            "source_portfolio_id": "portfolio:A",
        }
        with prior_art_review_audit_scope(
            assessment_kind="alpha6_targeted_reassessment",
            gap_id="gap:1",
        ):
            assert current_prior_art_review_audit_context() == {
                "assessment_kind": "alpha6_targeted_reassessment",
                "source_portfolio_id": "portfolio:A",
                "gap_id": "gap:1",
            }
        assert current_prior_art_review_audit_context() == {
            "assessment_kind": "alpha5_initial",
            "source_portfolio_id": "portfolio:A",
        }
    assert current_prior_art_review_audit_context() == {}


def test_summary_counts_exact_repeats_and_non_focal_waste():
    rows = [
        _row(
            "req:A",
            "resp:A",
            kind="alpha5_initial",
            hypothesis="h1",
        ),
        _row(
            "req:B",
            "resp:B",
            kind="alpha5_initial",
            hypothesis="h2",
        ),
        _row(
            "req:A",
            "resp:A",
            kind="alpha6_targeted_reassessment",
            hypothesis="h1",
            focal="h2",
        ),
        _row(
            "req:B",
            "resp:B",
            kind="alpha6_targeted_reassessment",
            hypothesis="h2",
            focal="h2",
        ),
    ]
    summary = summarize_prior_art_review_audit(rows)
    assert summary["calls"] == 4
    assert summary["unique_request_fingerprints"] == 2
    assert summary["duplicate_calls"] == 2
    assert summary["duplicate_cost_credits"] == 2.0
    assert summary["targeted_reassessment_calls"] == 2
    assert summary["targeted_reassessment_non_focal_calls"] == 1
    assert (
        summary[
            "targeted_reassessment_non_focal_duplicate_calls"
        ]
        == 1
    )
    assert summary["response_divergent_duplicate_groups"] == 0


def test_summary_flags_response_divergence_instead_of_calling_it_safe():
    rows = [
        _row(
            "req:A",
            "resp:first",
            kind="alpha5_initial",
            hypothesis="h1",
        ),
        _row(
            "req:A",
            "resp:second",
            kind="alpha6_targeted_reassessment",
            hypothesis="h1",
            focal="h1",
        ),
    ]
    summary = summarize_prior_art_review_audit(rows)
    assert summary["duplicate_calls"] == 1
    assert summary["response_stable_duplicate_groups"] == 0
    assert summary["response_divergent_duplicate_groups"] == 1
    assert summary["divergent_requests"][0]["claim_id"] is None
