from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExperimentalCheckType = str
ExperimentalCheckStatus = Literal["pass", "conditional", "fail", "unknown"]
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
RequirementCategory = str


class ExperimentalRequirement(StrictModel):
    requirement_id: str
    category: RequirementCategory
    capability: str
    necessity: RequirementNecessity
    rationale: str
    scientific_domain: str = "unknown"


class ExperimentalCheckResult(StrictModel):
    check_id: str
    hypothesis_id: str
    check_type: ExperimentalCheckType
    status: ExperimentalCheckStatus
    rationale: str
    scientific_domain: str = "unknown"

    precedent_status: PrecedentStatus | None = None
    evidence_paper_ids: list[str] = Field(default_factory=list)
    requirements: list[ExperimentalRequirement] = Field(default_factory=list)
    complexity: ComplexityLevel | None = None
    cost_burden: CostBurden | None = None
    effort_burden: EffortBurden | None = None


class ExperimentalRealizabilityReport(StrictModel):
    schema_version: Literal[
        "experimental-realizability-report-v02",
        "experimental-realizability-report-v03",
    ] = "experimental-realizability-report-v02"
    report_id: str
    source_intake_id: str
    source_intake_sha256: str
    source_physics_report_id: str
    source_scope_id: str
    source_validation_specification_id: str
    hypothesis_id: str
    scientific_domain: str = "unknown"

    disposition: ExperimentalDisposition
    checks: list[ExperimentalCheckResult] = Field(default_factory=list)

    # Existing fields remain for artifact/viewer compatibility. Domain adapters
    # may leave non-applicable concepts as unknown.
    synthesis_feasibility: ExperimentalCheckStatus = "unknown"
    structural_verifiability: ExperimentalCheckStatus = "unknown"
    active_site_verifiability: ExperimentalCheckStatus = "unknown"
    performance_testability: ExperimentalCheckStatus = "unknown"

    precedent_status: PrecedentStatus = "not_assessed"
    required_synthesis_capabilities: list[str] = Field(default_factory=list)
    required_characterization: list[str] = Field(default_factory=list)
    required_performance_tests: list[str] = Field(default_factory=list)
    required_measurement_capabilities: list[str] = Field(default_factory=list)

    # Legacy HER-specific alias.
    required_electrochemical_tests: list[str] = Field(default_factory=list)
    not_applicable_capabilities: list[str] = Field(default_factory=list)

    synthesis_complexity: ComplexityLevel = "unknown"
    characterization_complexity: ComplexityLevel = "unknown"
    relative_cost_burden: CostBurden = "unknown"
    relative_effort_burden: EffortBurden = "unknown"
    dominant_uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def synchronize_legacy_performance_alias(self) -> "ExperimentalRealizabilityReport":
        # Legacy HER report -> generic performance tests is safe. The reverse is
        # intentionally not automatic because a SERS measurement is not an
        # electrochemical test.
        if not self.required_performance_tests and self.required_electrochemical_tests:
            self.required_performance_tests = list(self.required_electrochemical_tests)
        if not self.required_measurement_capabilities:
            self.required_measurement_capabilities = sorted(set(
                self.required_characterization + self.required_performance_tests
            ))
        return self
