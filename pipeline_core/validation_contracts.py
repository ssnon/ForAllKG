from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


KnownValidationStrategy = Literal[
    "candidate_specific_computation",
    "comparative_computational_study",
    "mechanism_validation",
    "context_comparison",
    "literature_first",
    "hybrid",
]
ValidationStrategy = str


class ValidationSpecification(StrictModel):
    schema_version: Literal[
        "validation-specification-v02",
        "validation-specification-v03",
    ] = "validation-specification-v02"
    specification_id: str
    hypothesis_id: str
    source_scope_id: str

    scientific_domain: str = "unknown"
    validation_strategy: ValidationStrategy
    requires_candidate_concretization: bool

    controlled_variables: list[str] = Field(default_factory=list)
    varied_variables: list[str] = Field(default_factory=list)
    primary_observables: list[str] = Field(default_factory=list)
    secondary_observables: list[str] = Field(default_factory=list)
    required_comparisons: list[str] = Field(default_factory=list)
    candidate_concretization_requirements: list[str] = Field(default_factory=list)

    # Generic alpha3 names.
    required_scientific_checks: list[str] = Field(default_factory=list)
    not_applicable_scientific_checks: list[str] = Field(default_factory=list)
    required_experimental_capabilities: list[str] = Field(default_factory=list)

    # Legacy v0.2 aliases preserved for HER runtimes/artifacts.
    required_physics_checks: list[str] = Field(default_factory=list)
    not_applicable_physics_checks: list[str] = Field(default_factory=list)
    not_applicable_experimental_capabilities: list[str] = Field(default_factory=list)

    success_patterns: list[str] = Field(default_factory=list)
    falsification_patterns: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def synchronize_check_aliases(self) -> "ValidationSpecification":
        if not self.required_scientific_checks and self.required_physics_checks:
            self.required_scientific_checks = list(self.required_physics_checks)
        if not self.required_physics_checks and self.required_scientific_checks:
            self.required_physics_checks = list(self.required_scientific_checks)
        if (
            not self.not_applicable_scientific_checks
            and self.not_applicable_physics_checks
        ):
            self.not_applicable_scientific_checks = list(
                self.not_applicable_physics_checks
            )
        if (
            not self.not_applicable_physics_checks
            and self.not_applicable_scientific_checks
        ):
            self.not_applicable_physics_checks = list(
                self.not_applicable_scientific_checks
            )
        return self
