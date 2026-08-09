from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExperimentalCheckType = Literal[
    "synthesis_precedent",
    "synthesis_route_plausibility",
    "precursor_availability",
    "synthesis_complexity",
    "structural_verifiability",
    "active_site_verifiability",
    "performance_testability",
    "safety",
    "relative_cost_burden",
    "relative_effort_burden",
]

ExperimentalCheckStatus = Literal[
    "pass",
    "conditional",
    "fail",
    "unknown",
]

ExperimentalDisposition = Literal[
    "experimentally_plausible",
    "conditionally_plausible",
    "high_complexity",
    "experimentally_implausible",
    "insufficient_information",
    "human_review_required",
]

PrecedentStatus = Literal[
    "direct_precedent",
    "analogous_precedent",
    "no_precedent_found",
    "conflicting_precedent",
    "not_assessed",
]

ComplexityLevel = Literal["low", "moderate", "high", "very_high", "unknown"]
CostBurden = Literal["low", "moderate", "high", "very_high", "unknown"]
EffortBurden = Literal["low", "moderate", "high", "very_high", "unknown"]
RequirementNecessity = Literal["required", "recommended"]
RequirementCategory = Literal[
    "synthesis",
    "characterization",
    "electrochemistry",
    "safety",
]


class ExperimentalRequirement(StrictModel):
    requirement_id: str
    category: RequirementCategory
    capability: str
    necessity: RequirementNecessity
    rationale: str


class ExperimentalCheckResult(StrictModel):
    check_id: str
    hypothesis_id: str
    check_type: ExperimentalCheckType
    status: ExperimentalCheckStatus
    rationale: str

    precedent_status: PrecedentStatus | None = None
    evidence_paper_ids: list[str] = Field(default_factory=list)
    requirements: list[ExperimentalRequirement] = Field(default_factory=list)
    complexity: ComplexityLevel | None = None
    cost_burden: CostBurden | None = None
    effort_burden: EffortBurden | None = None


class ExperimentalRealizabilityReport(StrictModel):
    schema_version: Literal["experimental-realizability-report-v02"] = (
        "experimental-realizability-report-v02"
    )
    report_id: str
    source_intake_id: str
    source_intake_sha256: str
    source_physics_report_id: str
    source_scope_id: str
    source_validation_specification_id: str
    hypothesis_id: str

    disposition: ExperimentalDisposition
    checks: list[ExperimentalCheckResult] = Field(default_factory=list)

    synthesis_feasibility: ExperimentalCheckStatus
    structural_verifiability: ExperimentalCheckStatus
    active_site_verifiability: ExperimentalCheckStatus
    performance_testability: ExperimentalCheckStatus

    precedent_status: PrecedentStatus
    required_synthesis_capabilities: list[str] = Field(default_factory=list)
    required_characterization: list[str] = Field(default_factory=list)
    required_electrochemical_tests: list[str] = Field(default_factory=list)
    not_applicable_capabilities: list[str] = Field(default_factory=list)

    synthesis_complexity: ComplexityLevel
    characterization_complexity: ComplexityLevel
    relative_cost_burden: CostBurden
    relative_effort_burden: EffortBurden
    dominant_uncertainties: list[str] = Field(default_factory=list)
