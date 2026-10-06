from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.sers.validation_routing_contracts import SERSValidationRouteKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSRouteEvidenceRole = Literal[
    "generation_evidence",
    "independent_validation",
    "operator_observation",
    "experimental_result",
]

SERSRouteEvidenceReadiness = Literal[
    "raw",
    "screened",
    "review_ready",
]

SERSRouteEvidenceRelation = Literal[
    "supportive",
    "inconsistent",
    "mixed",
    "indeterminate",
    "not_assessed",
]

SERSRouteValidationStateName = Literal[
    "no_evidence",
    "evidence_available_not_reviewable",
    "review_ready",
    "deferred",
    "requires_interpretation",
]

SERSMechanismEvidenceState = Literal[
    "not_claimed",
    "incomplete",
    "review_ready",
]

SERSOutcomeEvidenceState = Literal[
    "incomplete",
    "review_ready",
]

SERSOverallResearchState = Literal[
    "requires_interpretation",
    "mechanism_evidence_incomplete",
    "outcome_evidence_incomplete",
    "routed_evidence_review_ready",
]


class SERSRouteEvidenceRecord(StrictModel):
    """Structured evidence supplied to a non-authoritative validator route.

    This contract deliberately records what an upstream validator or curator
    reports without granting hypothesis-verdict authority.  In particular,
    relation_to_claim is descriptive metadata; the orchestration layer does not
    aggregate it into support/contradiction of the whole hypothesis.
    """

    schema_version: Literal["sers-route-evidence-record-v0"] = (
        "sers-route-evidence-record-v0"
    )

    evidence_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    route_kind: SERSValidationRouteKind

    evidence_role: SERSRouteEvidenceRole
    review_readiness: SERSRouteEvidenceReadiness = "raw"
    relation_to_claim: SERSRouteEvidenceRelation = "not_assessed"

    target_observables: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    source_overlap_with_generation: bool | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance_notes: list[str] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _consistency(self) -> "SERSRouteEvidenceRecord":
        if self.route_kind == "unresolved":
            raise ValueError("unresolved routes cannot receive scientific evidence records")
        for values, label in (
            (self.target_observables, "target_observables"),
            (self.source_ids, "source_ids"),
            (self.source_paper_ids, "source_paper_ids"),
            (self.limitations, "limitations"),
            (self.provenance_notes, "provenance_notes"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SERSRouteEvidenceBundle(StrictModel):
    schema_version: Literal["sers-route-evidence-bundle-v0"] = (
        "sers-route-evidence-bundle-v0"
    )
    bundle_id: str = Field(min_length=1)
    records: list[SERSRouteEvidenceRecord] = Field(default_factory=list)
    record_count: int

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSRouteEvidenceBundle":
        if self.record_count != len(self.records):
            raise ValueError("record_count does not match records")
        ids = [row.evidence_id for row in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique")
        return self


class SERSIntegratedExperimentRequirement(StrictModel):
    """A non-procedural requirement set for an integrated SERS experiment.

    It specifies what must be observed/compared to resolve a routed hypothesis
    outcome.  It intentionally does not generate a wet-lab protocol, instrument
    settings, sample sizes, or fabrication recipes.
    """

    schema_version: Literal["sers-integrated-experiment-requirement-v0"] = (
        "sers-integrated-experiment-requirement-v0"
    )

    requirement_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_validation_plan_id: str = Field(min_length=1)
    source_validation_route_id: str = Field(min_length=1)

    route_target_observables: list[str] = Field(default_factory=list)
    source_prediction_ids: list[str] = Field(default_factory=list)
    source_falsification_criterion_ids: list[str] = Field(default_factory=list)

    comparison_requirements: list[str] = Field(default_factory=list)
    control_principles: list[str] = Field(default_factory=list)
    unresolved_protocol_dimensions: list[str] = Field(default_factory=list)

    protocol_generation_permitted: Literal[False] = False
    experiment_execution_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    hypothesis_verdict_authority: Literal[False] = False

    shadow_only: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _unique_lists(self) -> "SERSIntegratedExperimentRequirement":
        for values, label in (
            (self.route_target_observables, "route_target_observables"),
            (self.source_prediction_ids, "source_prediction_ids"),
            (
                self.source_falsification_criterion_ids,
                "source_falsification_criterion_ids",
            ),
            (self.comparison_requirements, "comparison_requirements"),
            (self.control_principles, "control_principles"),
            (self.unresolved_protocol_dimensions, "unresolved_protocol_dimensions"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SERSRouteValidationState(StrictModel):
    route_id: str = Field(min_length=1)
    route_kind: SERSValidationRouteKind
    required_for_outcome_assessment: bool = False
    required_for_mechanism_assessment: bool = False

    state: SERSRouteValidationStateName
    evidence_ids: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    high_cost_escalation_deferred: bool = False
    experiment_requirement_id: str | None = None

    component_scientific_verdict_permitted: Literal[False] = False
    whole_hypothesis_verdict_permitted: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False

    @model_validator(mode="after")
    def _unique_lists(self) -> "SERSRouteValidationState":
        for values, label in (
            (self.evidence_ids, "evidence_ids"),
            (self.blockers, "blockers"),
            (self.next_actions, "next_actions"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SERSHypothesisValidationState(StrictModel):
    schema_version: Literal["sers-hypothesis-validation-state-v0"] = (
        "sers-hypothesis-validation-state-v0"
    )

    state_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_validation_plan_id: str = Field(min_length=1)

    route_states: list[SERSRouteValidationState] = Field(default_factory=list)
    mechanism_evidence_state: SERSMechanismEvidenceState
    outcome_evidence_state: SERSOutcomeEvidenceState
    overall_research_state: SERSOverallResearchState

    recommended_next_route_kind: SERSValidationRouteKind | None = None
    recommended_next_route_id: str | None = None
    recommended_next_action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    experiment_requirement_ids: list[str] = Field(default_factory=list)

    scientific_verdict_permitted: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False

    shadow_only: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _consistency(self) -> "SERSHypothesisValidationState":
        route_ids = [row.route_id for row in self.route_states]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route state route_id values must be unique")
        if len(self.experiment_requirement_ids) != len(set(self.experiment_requirement_ids)):
            raise ValueError("experiment_requirement_ids must be unique")
        if (self.recommended_next_route_id is None) != (
            self.recommended_next_route_kind is None
        ):
            raise ValueError(
                "recommended_next_route_id and recommended_next_route_kind must be set together"
            )
        return self


class SERSValidationOrchestrationBundle(StrictModel):
    schema_version: Literal["sers-validation-orchestration-bundle-v0"] = (
        "sers-validation-orchestration-bundle-v0"
    )

    bundle_id: str = Field(min_length=1)
    source_portfolio_id: str = Field(min_length=1)
    source_validation_plan_bundle_id: str = Field(min_length=1)
    source_classical_em_readiness_bundle_id: str | None = None
    source_route_evidence_bundle_id: str | None = None

    states: list[SERSHypothesisValidationState] = Field(default_factory=list)
    state_count: int
    experiment_requirements: list[SERSIntegratedExperimentRequirement] = Field(
        default_factory=list
    )
    experiment_requirement_count: int

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    scientific_verdict_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSValidationOrchestrationBundle":
        if self.state_count != len(self.states):
            raise ValueError("state_count does not match states")
        if self.experiment_requirement_count != len(self.experiment_requirements):
            raise ValueError(
                "experiment_requirement_count does not match experiment_requirements"
            )
        state_ids = [row.state_id for row in self.states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("state_id values must be unique")
        requirement_ids = [row.requirement_id for row in self.experiment_requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("experiment requirement IDs must be unique")
        return self
