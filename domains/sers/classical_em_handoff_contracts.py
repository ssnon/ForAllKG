from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSClassicalEMHandoffStatus = Literal[
    "compiled_runnable_shadow",
    "compiled_requires_concretization",
    "compiled_unsupported",
    "compiled_not_applicable",
    "compiled_requires_interpretation",
    "compiled_invalid",
]


class SERSClassicalEMHandoff(StrictModel):
    schema_version: Literal[
        "sers-classical-em-handoff-v0"
    ] = "sers-classical-em-handoff-v0"

    handoff_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_validation_plan_id: str = Field(min_length=1)
    source_validation_route_id: str = Field(min_length=1)
    source_fdtd_applicability_report_id: str = Field(min_length=1)
    source_fdtd_subclaim_ids: list[str] = Field(min_length=1)
    route_scoped_source_fields: list[str] = Field(min_length=1)

    backend_family: Literal["classical_em"] = "classical_em"
    selected_backend: Literal["fdtd"] = "fdtd"
    backend_role: Literal["em_evidence_generator"] = "em_evidence_generator"

    source_simulation_spec_id: str = Field(min_length=1)
    source_simulation_validation_report_id: str = Field(min_length=1)
    simulation_disposition: Literal[
        "runnable_shadow",
        "requires_concretization",
        "unsupported",
        "not_applicable",
        "requires_interpretation",
        "invalid",
    ]
    handoff_status: SERSClassicalEMHandoffStatus

    required_for_outcome_assessment: bool = False
    required_for_mechanism_assessment: bool = False

    semantic_scope: Literal["electromagnetic_subclaim_only"] = (
        "electromagnetic_subclaim_only"
    )
    route_scoped_compilation: Literal[True] = True
    whole_hypothesis_verdict_permitted: Literal[False] = False
    integrated_sers_outcome_verdict_permitted: Literal[False] = False

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _unique_lineage(self) -> "SERSClassicalEMHandoff":
        if len(self.source_fdtd_subclaim_ids) != len(set(self.source_fdtd_subclaim_ids)):
            raise ValueError("source_fdtd_subclaim_ids must be unique")
        if len(self.route_scoped_source_fields) != len(set(self.route_scoped_source_fields)):
            raise ValueError("route_scoped_source_fields must be unique")
        return self


class SERSClassicalEMHandoffBundle(StrictModel):
    schema_version: Literal[
        "sers-classical-em-handoff-bundle-v0"
    ] = "sers-classical-em-handoff-bundle-v0"

    bundle_id: str = Field(min_length=1)
    source_portfolio_id: str = Field(min_length=1)
    source_validation_plan_bundle_id: str = Field(min_length=1)
    source_fdtd_applicability_bundle_id: str = Field(min_length=1)
    source_simulation_compilation_bundle_id: str = Field(min_length=1)
    source_simulation_validation_bundle_id: str = Field(min_length=1)
    domain_profile_id: Literal["sers_au_ag"] = "sers_au_ag"

    handoffs: list[SERSClassicalEMHandoff] = Field(default_factory=list)
    routed_hypothesis_count: int
    not_routed_hypothesis_ids: list[str] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSClassicalEMHandoffBundle":
        if self.routed_hypothesis_count != len(self.handoffs):
            raise ValueError("routed_hypothesis_count does not match handoffs")
        routed = [row.hypothesis_id for row in self.handoffs]
        if len(routed) != len(set(routed)):
            raise ValueError("at most one classical-EM handoff is allowed per hypothesis")
        if set(routed) & set(self.not_routed_hypothesis_ids):
            raise ValueError("hypothesis cannot be both routed and not routed")
        return self
