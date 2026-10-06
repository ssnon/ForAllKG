from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSValidationRouteKind = Literal[
    "classical_em",
    "molecular_spectroscopy",
    "surface_chemistry",
    "fabrication_process",
    "integrated_sers_experiment",
    "unresolved",
]

SERSValidationRouteReadiness = Literal[
    "ready",
    "deferred",
    "requires_interpretation",
]

SERSValidationContextTopic = Literal[
    "classical_em",
    "molecular_spectroscopy",
    "surface_chemistry",
    "fabrication_process",
    "integrated_sers_experiment",
]


class SERSValidationContextNote(StrictModel):
    context_id: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    text: str = Field(min_length=1)
    matched_topics: list[SERSValidationContextTopic] = Field(default_factory=list)
    routing_effect: Literal["context_only"] = "context_only"


class SERSValidationRoute(StrictModel):
    route_id: str = Field(min_length=1)
    route_kind: SERSValidationRouteKind
    statement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_fields: list[str] = Field(default_factory=list)
    target_observables: list[str] = Field(default_factory=list)
    validator_candidates: list[str] = Field(default_factory=list)
    readiness: SERSValidationRouteReadiness
    source_fdtd_subclaim_ids: list[str] = Field(default_factory=list)
    required_for_outcome_assessment: bool = False
    required_for_mechanism_assessment: bool = False
    # Compatibility mirror for v0 consumers. It is derived from the two
    # assessment roles above and must not be interpreted as a single-stage gate.
    required_for_whole_hypothesis_assessment: bool = False

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False

    @model_validator(mode="after")
    def _synchronize_assessment_roles(self) -> "SERSValidationRoute":
        self.required_for_whole_hypothesis_assessment = bool(
            self.required_for_whole_hypothesis_assessment
            or self.required_for_outcome_assessment
            or self.required_for_mechanism_assessment
        )
        return self


class SERSHypothesisValidationPlan(StrictModel):
    schema_version: Literal[
        "sers-hypothesis-validation-plan-v0",
        "sers-hypothesis-validation-plan-v01",
    ] = "sers-hypothesis-validation-plan-v01"

    plan_id: str = Field(min_length=1)
    source_portfolio_id: str = Field(min_length=1)
    source_fdtd_applicability_report_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    domain_profile_id: Literal["sers_au_ag"] = "sers_au_ag"

    fdtd_applicability: Literal[
        "direct",
        "partial",
        "not_applicable",
        "requires_interpretation",
    ]
    routes: list[SERSValidationRoute] = Field(default_factory=list)
    context_notes: list[SERSValidationContextNote] = Field(default_factory=list)
    mechanism_claim_present: bool = False
    outcome_assessment: Literal[
        "deferred_pending_routed_evidence"
    ] = "deferred_pending_routed_evidence"
    mechanism_assessment: Literal[
        "not_claimed",
        "deferred_pending_routed_evidence",
    ] = "not_claimed"
    whole_hypothesis_assessment: Literal[
        "deferred_pending_routed_evidence"
    ] = "deferred_pending_routed_evidence"
    integrated_experiment_required: bool

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _route_consistency(self) -> "SERSHypothesisValidationPlan":
        if not self.routes:
            raise ValueError("validation plan requires at least one route")
        route_ids = [row.route_id for row in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("validation route_id values must be unique")
        experiment_routes = [
            row
            for row in self.routes
            if row.route_kind == "integrated_sers_experiment"
        ]
        if self.integrated_experiment_required and not experiment_routes:
            raise ValueError(
                "integrated_experiment_required requires an integrated SERS route"
            )
        return self


class SERSHypothesisValidationPlanBundle(StrictModel):
    schema_version: Literal[
        "sers-hypothesis-validation-plan-bundle-v0",
        "sers-hypothesis-validation-plan-bundle-v01",
    ] = "sers-hypothesis-validation-plan-bundle-v01"

    bundle_id: str = Field(min_length=1)
    source_portfolio_id: str = Field(min_length=1)
    source_fdtd_applicability_bundle_id: str = Field(min_length=1)
    domain_profile_id: Literal["sers_au_ag"] = "sers_au_ag"
    plans: list[SERSHypothesisValidationPlan] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
