from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CatalystClass = Literal[
    "dual_atom",
    "single_atom",
    "mixed_atomic_site",
    "general_atomic_site",
    "unknown",
]

HypothesisLevel = Literal[
    "candidate_specific",
    "material_family",
    "mechanism",
    "comparative_study",
    "context_extension",
    "unknown",
]

ReactionClass = Literal["HER", "unknown"]
ScopeConfidence = Literal["high", "medium", "low"]


class ScientificScope(StrictModel):
    schema_version: Literal["scientific-scope-v02"] = "scientific-scope-v02"
    scope_id: str
    hypothesis_id: str

    catalyst_class: CatalystClass
    hypothesis_level: HypothesisLevel
    reaction: ReactionClass
    environments: list[str] = Field(default_factory=list)

    metals: list[str] = Field(default_factory=list)
    coordination_variables: list[str] = Field(default_factory=list)
    independent_variables: list[str] = Field(default_factory=list)
    dependent_observables: list[str] = Field(default_factory=list)

    requires_candidate_concretization: bool
    scope_confidence: ScopeConfidence
    scope_warnings: list[str] = Field(default_factory=list)

    catalyst_class_rationale: str
    hypothesis_level_rationale: str
