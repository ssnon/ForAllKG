from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ValidationStrategy = Literal[
    "candidate_specific_computation",
    "comparative_computational_study",
    "mechanism_validation",
    "context_comparison",
    "literature_first",
    "hybrid",
]


class ValidationSpecification(StrictModel):
    schema_version: Literal["validation-specification-v02"] = (
        "validation-specification-v02"
    )
    specification_id: str
    hypothesis_id: str
    source_scope_id: str

    validation_strategy: ValidationStrategy
    requires_candidate_concretization: bool

    controlled_variables: list[str] = Field(default_factory=list)
    varied_variables: list[str] = Field(default_factory=list)
    primary_observables: list[str] = Field(default_factory=list)
    secondary_observables: list[str] = Field(default_factory=list)
    required_comparisons: list[str] = Field(default_factory=list)
    candidate_concretization_requirements: list[str] = Field(default_factory=list)

    required_physics_checks: list[str] = Field(default_factory=list)
    not_applicable_physics_checks: list[str] = Field(default_factory=list)
    not_applicable_experimental_capabilities: list[str] = Field(default_factory=list)

    success_patterns: list[str] = Field(default_factory=list)
    falsification_patterns: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
