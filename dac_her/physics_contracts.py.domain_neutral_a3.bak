from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


PhysicsCheckType = Literal[
    "structural_validity",
    "pair_stability",
    "isolated_site_stability",
    "aggregation_risk",
    "thermodynamic_stability",
    "operating_state_stability",
    "hydrogen_adsorption",
    "water_dissociation",
    "oh_binding",
    "reaction_pathway",
    "electronic_structure",
]

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
    relevant_terms: list[str] = Field(default_factory=list)


class PhysicsCheckResult(StrictModel):
    check_id: str
    request_id: str
    hypothesis_id: str
    check_type: PhysicsCheckType
    status: PhysicsCheckStatus
    basis: PhysicsCheckBasis
    rationale: str

    value: float | None = None
    unit: str | None = None
    uncertainty: float | None = None

    source_statement_ids: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    tool_run_ids: list[str] = Field(default_factory=list)


class PhysicsFeasibilityReport(StrictModel):
    schema_version: Literal["physics-feasibility-report-v02"] = (
        "physics-feasibility-report-v02"
    )
    report_id: str
    source_intake_id: str
    source_intake_sha256: str
    source_scope_id: str
    source_validation_specification_id: str
    hypothesis_id: str
    disposition: PhysicsDisposition
    confidence: Literal["high", "medium", "low"]

    checks: list[PhysicsCheckResult] = Field(default_factory=list)
    blocking_checks: list[PhysicsCheckType] = Field(default_factory=list)
    unresolved_checks: list[PhysicsCheckType] = Field(default_factory=list)
    not_applicable_checks: list[PhysicsCheckType] = Field(default_factory=list)
    next_required_computations: list[str] = Field(default_factory=list)
