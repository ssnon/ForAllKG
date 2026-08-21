from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Alpha3: check identifiers are domain-extensible. The existing HER identifiers
# remain valid strings and their runtime behavior is unchanged.
PhysicsCheckType = str

PhysicsCheckStatus = Literal[
    "pass",
    "conditional",
    "fail",
    "unknown",
    "requires_computation",
]

PhysicsCheckBasis = Literal[
    "deterministic_rule",
    "reported_evidence",
    "computed_value",
    "surrogate_prediction",
    "unavailable",
]

PhysicsDisposition = Literal[
    "physically_supported",
    "conditionally_supported",
    "requires_computation",
    "physically_implausible",
    "insufficient_information",
    "human_review_required",
]


class PhysicsCheckRequest(StrictModel):
    request_id: str
    hypothesis_id: str
    source_scope_id: str
    source_validation_specification_id: str
    check_type: PhysicsCheckType
    reason: str
    scientific_domain: str = "unknown"
    backend_class: str | None = None
    relevant_terms: list[str] = Field(default_factory=list)


class PhysicsCheckResult(StrictModel):
    check_id: str
    request_id: str
    hypothesis_id: str
    check_type: PhysicsCheckType
    status: PhysicsCheckStatus
    basis: PhysicsCheckBasis
    rationale: str
    scientific_domain: str = "unknown"
    backend_class: str | None = None

    value: float | None = None
    unit: str | None = None
    uncertainty: float | None = None

    source_statement_ids: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    tool_run_ids: list[str] = Field(default_factory=list)


class PhysicsFeasibilityReport(StrictModel):
    schema_version: Literal[
        "physics-feasibility-report-v02",
        "physics-feasibility-report-v03",
    ] = "physics-feasibility-report-v02"
    report_id: str
    source_intake_id: str
    source_intake_sha256: str
    source_scope_id: str
    source_validation_specification_id: str
    hypothesis_id: str
    scientific_domain: str = "unknown"
    disposition: PhysicsDisposition
    confidence: Literal["high", "medium", "low"]

    checks: list[PhysicsCheckResult] = Field(default_factory=list)
    blocking_checks: list[str] = Field(default_factory=list)
    unresolved_checks: list[str] = Field(default_factory=list)
    not_applicable_checks: list[str] = Field(default_factory=list)
    next_required_computations: list[str] = Field(default_factory=list)
