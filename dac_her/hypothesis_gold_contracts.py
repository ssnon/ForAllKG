from __future__ import annotations

from typing import Literal

from pydantic import Field

from dac_her.hypothesis_semantic_contracts import (
    SemanticDimension,
    SemanticVerdict,
    StrictModel,
)


class SemanticGoldExpectation(StrictModel):
    dimension: SemanticDimension
    allowed_verdicts: list[SemanticVerdict] = Field(min_length=1)
    critical: bool = True
    note: str = ""


class SemanticGoldCase(StrictModel):
    case_id: str
    description: str
    context_path: str
    portfolio_path: str
    expectations: list[SemanticGoldExpectation] = Field(min_length=1)
    forbid_unexpected_failures: bool = True
    allowed_additional_fail_dimensions: list[SemanticDimension] = Field(
        default_factory=list
    )


class SemanticGoldSuite(StrictModel):
    schema_version: Literal["hypothesis-semantic-gold-suite-v262"] = (
        "hypothesis-semantic-gold-suite-v262"
    )
    suite_id: str
    cases: list[SemanticGoldCase]


class SemanticGoldMismatch(StrictModel):
    case_id: str
    dimension: SemanticDimension
    allowed_verdicts: list[SemanticVerdict]
    actual_verdict: SemanticVerdict | None
    critical: bool
    kind: Literal[
        "critical_mismatch",
        "noncritical_mismatch",
        "missing_review",
        "missing_dimension",
        "unexpected_failure",
    ]
    note: str = ""


class SemanticGoldCaseComparison(StrictModel):
    case_id: str
    passed: bool
    exact_or_allowed_agreements: int
    mismatches: list[SemanticGoldMismatch] = Field(default_factory=list)


class SemanticGoldComparisonReport(StrictModel):
    schema_version: Literal["hypothesis-semantic-gold-comparison-v262"] = (
        "hypothesis-semantic-gold-comparison-v262"
    )
    suite_id: str
    passed: bool
    case_count: int
    passed_cases: int
    failed_cases: int
    critical_mismatches: int
    noncritical_mismatches: int
    missing_reviews: int
    case_results: list[SemanticGoldCaseComparison]
