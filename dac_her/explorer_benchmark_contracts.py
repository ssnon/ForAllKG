from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


EpistemicRoleName = Literal["reported", "evidence_synthesis", "navigation_note", "unresolved"]
RepairExpectation = Literal["none", "allowed", "required"]


class StrictBenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkExpectations(StrictBenchmarkModel):
    must_validate: bool = True
    max_repairs: int = 1
    max_compile_issues: int = 0
    min_reported_statements: int = 0
    min_synthesis_statements: int = 0
    min_unresolved_connections: int = 0
    min_mechanism_routes: int = 0

    required_terms_anywhere: List[str] = Field(default_factory=list)
    required_terms_by_role: Dict[EpistemicRoleName, List[str]] = Field(default_factory=dict)
    forbidden_terms_by_role: Dict[EpistemicRoleName, List[str]] = Field(default_factory=dict)
    forbidden_regexes: List[str] = Field(default_factory=list)

    require_candidate_verification_propagation: bool = True
    forbid_partial_paper_absence_claims: bool = True
    forbid_alignment_causal_claims: bool = True
    require_report_packet_sha_match: bool = True

    repair_expectation: RepairExpectation = "allowed"
    max_repair_text_changes: Optional[int] = None
    require_support_only_repair: bool = False

    @model_validator(mode="after")
    def _bounds(self) -> "BenchmarkExpectations":
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be >= 0")
        for name in (
            "max_compile_issues",
            "min_reported_statements",
            "min_synthesis_statements",
            "min_unresolved_connections",
            "min_mechanism_routes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        return self


class BenchmarkCase(StrictBenchmarkModel):
    case_id: str
    description: str
    enabled: bool = True
    packet: str
    output_prefix: str
    expectations: BenchmarkExpectations = Field(default_factory=BenchmarkExpectations)
    tags: List[str] = Field(default_factory=list)


class BenchmarkSuite(StrictBenchmarkModel):
    schema_version: Literal["graph-explorer-benchmark-suite-v1"] = "graph-explorer-benchmark-suite-v1"
    suite_id: str
    description: str
    cases: List[BenchmarkCase]

    @model_validator(mode="after")
    def _unique_case_ids(self) -> "BenchmarkSuite":
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate benchmark case_id")
        return self


class BenchmarkIssue(StrictBenchmarkModel):
    severity: Literal["error", "warning"]
    code: str
    message: str


class BenchmarkCaseResult(StrictBenchmarkModel):
    case_id: str
    passes: bool
    skipped: bool = False
    issues: List[BenchmarkIssue] = Field(default_factory=list)
    metrics: Dict[str, object] = Field(default_factory=dict)


class BenchmarkSuiteResult(StrictBenchmarkModel):
    schema_version: Literal["graph-explorer-benchmark-result-v1"] = "graph-explorer-benchmark-result-v1"
    suite_id: str
    passes: bool
    evaluated_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    case_results: List[BenchmarkCaseResult]
