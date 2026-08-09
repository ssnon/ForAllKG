from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dac_her.hypothesis_gold_contracts import SemanticGoldExpectation
from dac_her.hypothesis_semantic_contracts import (
    SEMANTIC_DIMENSIONS,
    SemanticDimension,
    StrictModel,
)


def _validate_complete_expectations(
    expectations: list[SemanticGoldExpectation],
) -> None:
    names = [row.dimension for row in expectations]
    if len(names) != len(set(names)):
        raise ValueError("real gold contains duplicate semantic dimensions")
    missing = sorted(set(SEMANTIC_DIMENSIONS) - set(names))
    extra = sorted(set(names) - set(SEMANTIC_DIMENSIONS))
    if missing or extra or len(names) != len(SEMANTIC_DIMENSIONS):
        raise ValueError(
            "real gold must label all semantic dimensions exactly once; "
            f"missing={missing}, extra={extra}"
        )


class HypothesisRealGoldCaseSpec(StrictModel):
    case_id: str
    description: str
    context_path: str
    portfolio_path: str
    expectations: list[SemanticGoldExpectation] = Field(min_length=1)
    forbid_unexpected_failures: bool = True
    allowed_additional_fail_dimensions: list[SemanticDimension] = Field(
        default_factory=list
    )
    generator_version: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _complete_semantic_labels(self) -> "HypothesisRealGoldCaseSpec":
        _validate_complete_expectations(self.expectations)
        return self


class HypothesisRealGoldSpec(StrictModel):
    schema_version: Literal["hypothesis-real-gold-spec-v262"] = (
        "hypothesis-real-gold-spec-v262"
    )
    suite_id: str
    cases: list[HypothesisRealGoldCaseSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> "HypothesisRealGoldSpec":
        ids = [row.case_id for row in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("real gold spec contains duplicate case_id")
        return self


class HypothesisRealGoldArtifactLineage(StrictModel):
    context_file_sha256: str
    portfolio_file_sha256: str
    context_id: str
    context_declared_sha256: str
    portfolio_id: str
    portfolio_source_context_id: str
    portfolio_source_context_sha256: str
    context_source_report_id: str
    context_source_report_sha256: str
    portfolio_source_report_id: str
    portfolio_source_report_sha256: str


class HypothesisRealGoldCase(StrictModel):
    case_id: str
    description: str
    context_path: str
    portfolio_path: str
    lineage: HypothesisRealGoldArtifactLineage
    expectations: list[SemanticGoldExpectation] = Field(min_length=1)
    forbid_unexpected_failures: bool = True
    allowed_additional_fail_dimensions: list[SemanticDimension] = Field(
        default_factory=list
    )
    generator_version: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _complete_semantic_labels(self) -> "HypothesisRealGoldCase":
        _validate_complete_expectations(self.expectations)
        return self


class HypothesisRealGoldSuite(StrictModel):
    schema_version: Literal["hypothesis-real-gold-suite-v262"] = (
        "hypothesis-real-gold-suite-v262"
    )
    suite_id: str
    cases: list[HypothesisRealGoldCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> "HypothesisRealGoldSuite":
        ids = [row.case_id for row in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("real gold suite contains duplicate case_id")
        return self


RealGoldLineageIssueCode = Literal[
    "MISSING_CONTEXT_FILE",
    "MISSING_PORTFOLIO_FILE",
    "CONTEXT_FILE_SHA_MISMATCH",
    "PORTFOLIO_FILE_SHA_MISMATCH",
    "INVALID_CONTEXT_FILE",
    "INVALID_PORTFOLIO_FILE",
    "CONTEXT_ID_MISMATCH",
    "CONTEXT_DECLARED_SHA_MISMATCH",
    "PORTFOLIO_ID_MISMATCH",
    "PORTFOLIO_CONTEXT_ID_MISMATCH",
    "PORTFOLIO_CONTEXT_SHA_MISMATCH",
    "CONTEXT_REPORT_ID_MISMATCH",
    "CONTEXT_REPORT_SHA_MISMATCH",
    "PORTFOLIO_REPORT_ID_MISMATCH",
    "PORTFOLIO_REPORT_SHA_MISMATCH",
    "CROSS_ARTIFACT_CONTEXT_ID_MISMATCH",
    "CROSS_ARTIFACT_CONTEXT_SHA_MISMATCH",
    "CROSS_ARTIFACT_REPORT_ID_MISMATCH",
    "CROSS_ARTIFACT_REPORT_SHA_MISMATCH",
]


class HypothesisRealGoldLineageIssue(StrictModel):
    code: RealGoldLineageIssueCode
    message: str


class HypothesisRealGoldLineageCaseCheck(StrictModel):
    case_id: str
    passed: bool
    issues: list[HypothesisRealGoldLineageIssue] = Field(default_factory=list)


class HypothesisRealGoldLineagePreflightReport(StrictModel):
    schema_version: Literal["hypothesis-real-gold-lineage-preflight-v262"] = (
        "hypothesis-real-gold-lineage-preflight-v262"
    )
    suite_id: str
    passed: bool
    case_count: int
    failed_cases: int
    case_results: list[HypothesisRealGoldLineageCaseCheck]
