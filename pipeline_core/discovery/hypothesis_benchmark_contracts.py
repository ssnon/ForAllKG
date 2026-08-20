from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


IssueSeverity = Literal["error", "warning"]
IssueLayer = Literal["hard_gate", "diagnostic"]


class HypothesisBenchmarkIssue(StrictModel):
    severity: IssueSeverity
    layer: IssueLayer
    code: str
    location: str
    message: str
    hypothesis_id: str | None = None
    statement_ids: list[str] = Field(default_factory=list)
    source: str


class HypothesisEvaluationReport(StrictModel):
    schema_version: Literal["hypothesis-evaluation-report-v262"] = (
        "hypothesis-evaluation-report-v262"
    )
    evaluator_version: str
    context_id: str
    context_sha256: str
    portfolio_id: str
    portfolio_sha256: str
    hard_gate_passed: bool
    hard_gate_errors: int
    hard_gate_warnings: int
    hard_gate_issues: list[HypothesisBenchmarkIssue] = Field(default_factory=list)
    diagnostics: list[HypothesisBenchmarkIssue] = Field(default_factory=list)


class BenchmarkExpectation(StrictModel):
    hard_gate_pass: bool
    expected_abstention: bool | None = None
    required_issue_codes: list[str] = Field(default_factory=list)
    forbidden_issue_codes: list[str] = Field(default_factory=list)
    required_diagnostic_codes: list[str] = Field(default_factory=list)
    forbidden_diagnostic_codes: list[str] = Field(default_factory=list)


class HypothesisBenchmarkCase(StrictModel):
    case_id: str
    category: Literal["canonical", "adversarial"]
    description: str
    context_path: str
    portfolio_path: str
    expectation: BenchmarkExpectation


class HypothesisBenchmarkSuite(StrictModel):
    schema_version: Literal["hypothesis-benchmark-suite-v262"] = (
        "hypothesis-benchmark-suite-v262"
    )
    suite_id: str
    evaluator_version: str
    cases: list[HypothesisBenchmarkCase]


class HypothesisBenchmarkCaseResult(StrictModel):
    case_id: str
    category: Literal["canonical", "adversarial"]
    expectation_passed: bool
    expectation_failures: list[str] = Field(default_factory=list)
    report: HypothesisEvaluationReport


class HypothesisBenchmarkSuiteResult(StrictModel):
    schema_version: Literal["hypothesis-benchmark-suite-result-v262"] = (
        "hypothesis-benchmark-suite-result-v262"
    )
    suite_id: str
    evaluator_version: str
    passed: bool
    passed_cases: int
    failed_cases: int
    case_results: list[HypothesisBenchmarkCaseResult]
