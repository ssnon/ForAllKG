from __future__ import annotations

from dac_her.hypothesis_gold_comparator import HypothesisSemanticGoldComparator
from dac_her.hypothesis_gold_contracts import SemanticGoldSuite
from dac_her.hypothesis_semantic_contracts import (
    HypothesisSemanticDimensionDraft,
    HypothesisSemanticReview,
    SEMANTIC_DIMENSIONS,
)


def review(verdict_overrides=None):
    verdict_overrides = verdict_overrides or {}
    rows = [
        HypothesisSemanticDimensionDraft(
            dimension=dimension,
            verdict=verdict_overrides.get(
                dimension,
                "not_applicable" if dimension == "hypothesis_distinctness" else "pass",
            ),
            rationale="fixture",
        )
        for dimension in SEMANTIC_DIMENSIONS
    ]
    return HypothesisSemanticReview(
        review_id="review",
        source_context_id="ctx",
        source_context_sha256="csha",
        source_portfolio_id="portfolio",
        source_portfolio_sha256="psha",
        source_evaluator_version="eval",
        source_hard_gate_passed=True,
        critic_prompt_version="prompt",
        critic_prompt_sha256="promptsha",
        dimensions=rows,
        overall_summary="fixture",
    )


def suite(allowed, *, critical=True):
    return SemanticGoldSuite.model_validate(
        {
            "suite_id": "gold",
            "cases": [
                {
                    "case_id": "case",
                    "description": "fixture",
                    "context_path": "context.json",
                    "portfolio_path": "portfolio.json",
                    "expectations": [
                        {
                            "dimension": "directional_specificity",
                            "allowed_verdicts": allowed,
                            "critical": critical,
                        }
                    ],
                }
            ],
        }
    )


def test_gold_comparator_accepts_allowed_verdict():
    report = HypothesisSemanticGoldComparator().compare(
        suite(["warning"]),
        {"case": review({"directional_specificity": "warning"})},
    )
    assert report.passed
    assert report.critical_mismatches == 0


def test_gold_comparator_fails_critical_mismatch():
    report = HypothesisSemanticGoldComparator().compare(
        suite(["warning"]),
        {"case": review({"directional_specificity": "pass"})},
    )
    assert not report.passed
    assert report.critical_mismatches == 1


def test_gold_comparator_allows_noncritical_mismatch():
    report = HypothesisSemanticGoldComparator().compare(
        suite(["warning"], critical=False),
        {"case": review({"directional_specificity": "pass"})},
    )
    assert report.passed
    assert report.noncritical_mismatches == 1

def test_gold_comparator_rejects_unexpected_fail():
    r = review({
        "directional_specificity": "warning",
        "cross_paper_discipline": "fail",
    })

    report = HypothesisSemanticGoldComparator().compare(
        suite(["warning"]),
        {"case": r},
    )

    assert not report.passed
    assert any(
        m.kind == "unexpected_failure"
        for m in report.case_results[0].mismatches
    )

def test_gold_comparator_allows_unexpected_warning():
    r = review({
        "directional_specificity": "warning",
        "inferential_proportionality": "warning",
    })

    report = HypothesisSemanticGoldComparator().compare(
        suite(["warning"]),
        {"case": r},
    )

    assert report.passed