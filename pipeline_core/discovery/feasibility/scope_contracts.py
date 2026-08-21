from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Compatibility vocabulary retained for DAC-HER callers/documentation.
KnownCatalystClass = Literal[
    "dual_atom",
    "single_atom",
    "mixed_atomic_site",
    "general_atomic_site",
    "unknown",
]
KnownReactionClass = Literal["HER", "unknown"]

# Alpha3: contracts are no longer closed to one scientific domain.
CatalystClass = str
ReactionClass = str
HypothesisLevel = str
ScopeConfidence = Literal["high", "medium", "low"]


class ScientificScope(StrictModel):
    schema_version: Literal[
        "scientific-scope-v02",
        "scientific-scope-v03",
    ] = "scientific-scope-v02"
    scope_id: str
    hypothesis_id: str

    # Legacy v0.2 fields. They remain readable/writable so old HER artifacts and
    # downstream viewers do not break. New domains should prefer the generic
    # mirrors below and may leave these as "unknown" where they do not apply.
    catalyst_class: CatalystClass = "unknown"
    hypothesis_level: HypothesisLevel = "unknown"
    reaction: ReactionClass = "unknown"
    environments: list[str] = Field(default_factory=list)
    metals: list[str] = Field(default_factory=list)
    coordination_variables: list[str] = Field(default_factory=list)

    # Domain-neutral alpha3 fields.
    system_class: str = "unknown"
    scientific_domain: str = "unknown"
    process: str = "unknown"
    components: list[str] = Field(default_factory=list)
    structural_variables: list[str] = Field(default_factory=list)

    independent_variables: list[str] = Field(default_factory=list)
    dependent_observables: list[str] = Field(default_factory=list)

    requires_candidate_concretization: bool
    scope_confidence: ScopeConfidence
    scope_warnings: list[str] = Field(default_factory=list)

    catalyst_class_rationale: str = ""
    hypothesis_level_rationale: str
    system_class_rationale: str = ""

    @model_validator(mode="after")
    def synchronize_legacy_and_generic_fields(self) -> "ScientificScope":
        # Legacy -> generic is always safe because the generic field is broader.
        if self.system_class == "unknown" and self.catalyst_class != "unknown":
            self.system_class = self.catalyst_class
        if self.process == "unknown" and self.reaction != "unknown":
            self.process = self.reaction
        if not self.components and self.metals:
            self.components = list(self.metals)
        if not self.structural_variables and self.coordination_variables:
            self.structural_variables = list(self.coordination_variables)
        if not self.system_class_rationale and self.catalyst_class_rationale:
            self.system_class_rationale = self.catalyst_class_rationale

        # Generic -> legacy is deliberately narrow. Never call an arbitrary SERS
        # system a "catalyst" or an arbitrary process a "reaction".
        legacy_classes = {
            "dual_atom",
            "single_atom",
            "mixed_atomic_site",
            "general_atomic_site",
        }
        if self.catalyst_class == "unknown" and self.system_class in legacy_classes:
            self.catalyst_class = self.system_class
        if self.reaction == "unknown" and self.process == "HER":
            self.reaction = "HER"
        return self
