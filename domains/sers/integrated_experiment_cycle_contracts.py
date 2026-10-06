from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.sers.validation_orchestration_contracts import SERSRouteEvidenceBundle
from domains.sers.validation_routing_contracts import SERSValidationRouteKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSExperimentPlanStatus = Literal[
    "not_candidate",
    "review_candidate",
    "direct_test_review_candidate",
]

SERSExperimentInformationPriority = Literal[
    "not_prioritized",
    "candidate_after_human_review",
    "preferred_over_deferred_high_cost_em_escalation",
]

SERSExperimentResultRelation = Literal[
    "supportive",
    "inconsistent",
    "mixed",
    "indeterminate",
]

SERSExperimentResultQuality = Literal[
    "review_ready",
    "limited",
    "unresolved",
]

SERSExperimentControlsStatus = Literal[
    "adequate",
    "partial",
    "inadequate",
    "not_assessed",
]

SERSProtocolDeviationStatus = Literal[
    "none",
    "minor",
    "major",
    "not_assessed",
]

SERSResultAssessmentBasis = Literal[
    "predeclared_criterion_match",
    "expert_review",
    "statistical_analysis",
    "mixed",
]

SERSFailureAttributionCategory = Literal[
    "classical_em_model_or_mechanism",
    "molecular_or_surface_chemistry",
    "fabrication_realization",
    "analyte_accessibility_or_orientation",
    "measurement_or_protocol_confound",
    "interaction_nonadditivity_or_alternative_mechanism",
]

SERSFailureAttributionStatus = Literal[
    "candidate_explanation_only",
    "contextually_deprioritized_not_excluded",
    "requires_targeted_followup",
]

SERSForAllKGFeedbackKind = Literal[
    "preserve_outcome_support_with_mechanism_uncertainty",
    "revisit_component_mechanism",
    "investigate_alternative_mechanism",
    "inspect_fabrication_realization",
    "repeat_or_refine_integrated_experiment",
    "preserve_indeterminate_outcome",
]


class SERSExperimentComparisonDesign(StrictModel):
    focal_hypothesis_statement: str = Field(min_length=1)
    target_observables: list[str] = Field(default_factory=list)
    prediction_statements: list[str] = Field(default_factory=list)
    falsification_statements: list[str] = Field(default_factory=list)
    comparison_requirements: list[str] = Field(default_factory=list)
    control_principles: list[str] = Field(default_factory=list)
    unresolved_protocol_dimensions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_lists(self) -> "SERSExperimentComparisonDesign":
        for values, label in (
            (self.target_observables, "target_observables"),
            (self.prediction_statements, "prediction_statements"),
            (self.falsification_statements, "falsification_statements"),
            (self.comparison_requirements, "comparison_requirements"),
            (self.control_principles, "control_principles"),
            (self.unresolved_protocol_dimensions, "unresolved_protocol_dimensions"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SERSIntegratedExperimentReviewPlan(StrictModel):
    schema_version: Literal["sers-integrated-experiment-review-plan-v0"] = (
        "sers-integrated-experiment-review-plan-v0"
    )
    plan_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_scientific_review_id: str = Field(min_length=1)
    source_validation_state_id: str = Field(min_length=1)
    source_experiment_requirement_id: str = Field(min_length=1)
    source_validation_route_id: str = Field(min_length=1)

    plan_status: SERSExperimentPlanStatus
    information_priority: SERSExperimentInformationPriority
    comparison_design: SERSExperimentComparisonDesign
    result_interpretation_rules: list[str] = Field(default_factory=list)
    preserved_uncertainties: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    human_design_review_required: Literal[True] = True
    protocol_generation_permitted: Literal[False] = False
    experiment_execution_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    hypothesis_verdict_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _unique_lists(self) -> "SERSIntegratedExperimentReviewPlan":
        for values, label in (
            (self.result_interpretation_rules, "result_interpretation_rules"),
            (self.preserved_uncertainties, "preserved_uncertainties"),
            (self.blockers, "blockers"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SERSIntegratedExperimentReviewPlanBundle(StrictModel):
    schema_version: Literal["sers-integrated-experiment-review-plan-bundle-v0"] = (
        "sers-integrated-experiment-review-plan-bundle-v0"
    )
    bundle_id: str = Field(min_length=1)
    source_portfolio_id: str = Field(min_length=1)
    source_validation_plan_bundle_id: str = Field(min_length=1)
    source_orchestration_bundle_id: str = Field(min_length=1)
    source_scientific_review_bundle_id: str = Field(min_length=1)
    plans: list[SERSIntegratedExperimentReviewPlan] = Field(default_factory=list)
    plan_count: int

    shadow_only: Literal[True] = True
    experiment_execution_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    hypothesis_verdict_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSIntegratedExperimentReviewPlanBundle":
        if self.plan_count != len(self.plans):
            raise ValueError("plan_count does not match plans")
        ids = [row.plan_id for row in self.plans]
        if len(ids) != len(set(ids)):
            raise ValueError("plan_id values must be unique")
        return self


class SERSIntegratedExperimentResultSubmission(StrictModel):
    schema_version: Literal["sers-integrated-experiment-result-submission-v0"] = (
        "sers-integrated-experiment-result-submission-v0"
    )
    submission_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    experiment_plan_id: str = Field(min_length=1)
    experiment_requirement_id: str = Field(min_length=1)

    relation_to_claim: SERSExperimentResultRelation
    result_quality: SERSExperimentResultQuality
    assessment_basis: SERSResultAssessmentBasis
    measurement_summary: str = Field(min_length=1)
    measured_observables: list[str] = Field(default_factory=list)
    comparison_summary: str = Field(min_length=1)
    controls_status: SERSExperimentControlsStatus
    protocol_deviation_status: SERSProtocolDeviationStatus
    source_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "SERSIntegratedExperimentResultSubmission":
        if self.result_quality == "review_ready":
            if not self.source_ids:
                raise ValueError("review_ready experimental result requires source_ids")
            if self.controls_status in {"inadequate", "not_assessed"}:
                raise ValueError(
                    "review_ready experimental result requires adequate or partial controls"
                )
            if self.protocol_deviation_status in {"major", "not_assessed"}:
                raise ValueError(
                    "review_ready experimental result cannot have major/not-assessed protocol deviation"
                )
        for values, label in (
            (self.measured_observables, "measured_observables"),
            (self.source_ids, "source_ids"),
            (self.limitations, "limitations"),
            (self.provenance_notes, "provenance_notes"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SERSIntegratedExperimentResultSubmissionBundle(StrictModel):
    schema_version: Literal[
        "sers-integrated-experiment-result-submission-bundle-v0"
    ] = "sers-integrated-experiment-result-submission-bundle-v0"
    bundle_id: str = Field(min_length=1)
    submissions: list[SERSIntegratedExperimentResultSubmission] = Field(default_factory=list)
    submission_count: int

    @model_validator(mode="after")
    def _counts(self) -> "SERSIntegratedExperimentResultSubmissionBundle":
        if self.submission_count != len(self.submissions):
            raise ValueError("submission_count does not match submissions")
        ids = [row.submission_id for row in self.submissions]
        if len(ids) != len(set(ids)):
            raise ValueError("submission_id values must be unique")
        return self


class SERSIntegratedExperimentResultRecord(StrictModel):
    result_id: str = Field(min_length=1)
    source_submission_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_experiment_plan_id: str = Field(min_length=1)
    source_experiment_requirement_id: str = Field(min_length=1)
    source_validation_route_id: str = Field(min_length=1)

    relation_to_claim: SERSExperimentResultRelation
    result_quality: SERSExperimentResultQuality
    assessment_basis: SERSResultAssessmentBasis
    measurement_summary: str = Field(min_length=1)
    measured_observables: list[str] = Field(default_factory=list)
    comparison_summary: str = Field(min_length=1)
    controls_status: SERSExperimentControlsStatus
    protocol_deviation_status: SERSProtocolDeviationStatus
    source_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance_notes: list[str] = Field(default_factory=list)

    route_evidence_id: str = Field(min_length=1)
    review_ready_for_scientific_aggregation: bool

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSIntegratedExperimentResultBundle(StrictModel):
    schema_version: Literal["sers-integrated-experiment-result-bundle-v0"] = (
        "sers-integrated-experiment-result-bundle-v0"
    )
    bundle_id: str = Field(min_length=1)
    source_experiment_plan_bundle_id: str = Field(min_length=1)
    results: list[SERSIntegratedExperimentResultRecord] = Field(default_factory=list)
    result_count: int
    merged_route_evidence_bundle_id: str = Field(min_length=1)

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSIntegratedExperimentResultBundle":
        if self.result_count != len(self.results):
            raise ValueError("result_count does not match results")
        ids = [row.result_id for row in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("result_id values must be unique")
        return self


class SERSFailureAttributionCandidate(StrictModel):
    attribution_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    category: SERSFailureAttributionCategory
    status: SERSFailureAttributionStatus
    route_kind: SERSValidationRouteKind | None = None
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    next_test_question: str = Field(min_length=1)

    causal_conclusion_permitted: Literal[False] = False
    automatic_hypothesis_rejection_permitted: Literal[False] = False


class SERSForAllKGFeedbackCandidate(StrictModel):
    feedback_candidate_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    feedback_kind: SERSForAllKGFeedbackKind
    target_route_kind: SERSValidationRouteKind | None = None
    action: str = Field(min_length=1)
    research_question: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_result_ids: list[str] = Field(default_factory=list)
    source_review_id: str = Field(min_length=1)

    canonical_feedback_permitted: Literal[False] = False
    automatic_hypothesis_rewrite_permitted: Literal[False] = False
    automatic_hypothesis_rejection_permitted: Literal[False] = False
    requires_future_policy_or_human_review: Literal[True] = True


class SERSIntegratedExperimentFollowup(StrictModel):
    followup_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_scientific_review_id: str = Field(min_length=1)
    source_experiment_result_ids: list[str] = Field(default_factory=list)
    outcome_verdict: str = Field(min_length=1)
    mechanism_verdict: str = Field(min_length=1)
    failure_analysis_required: bool
    attribution_candidates: list[SERSFailureAttributionCandidate] = Field(default_factory=list)
    feedback_candidates: list[SERSForAllKGFeedbackCandidate] = Field(default_factory=list)
    interpretation: str = Field(min_length=1)

    whole_hypothesis_confirmation_permitted: Literal[False] = False
    automatic_hypothesis_rejection_permitted: Literal[False] = False
    canonical_feedback_permitted: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSIntegratedExperimentFollowupBundle(StrictModel):
    schema_version: Literal["sers-integrated-experiment-followup-bundle-v0"] = (
        "sers-integrated-experiment-followup-bundle-v0"
    )
    bundle_id: str = Field(min_length=1)
    source_scientific_review_bundle_id: str = Field(min_length=1)
    source_experiment_result_bundle_id: str = Field(min_length=1)
    followups: list[SERSIntegratedExperimentFollowup] = Field(default_factory=list)
    followup_count: int

    shadow_only: Literal[True] = True
    whole_hypothesis_confirmation_permitted: Literal[False] = False
    automatic_hypothesis_rejection_permitted: Literal[False] = False
    canonical_feedback_permitted: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSIntegratedExperimentFollowupBundle":
        if self.followup_count != len(self.followups):
            raise ValueError("followup_count does not match followups")
        ids = [row.followup_id for row in self.followups]
        if len(ids) != len(set(ids)):
            raise ValueError("followup_id values must be unique")
        return self
