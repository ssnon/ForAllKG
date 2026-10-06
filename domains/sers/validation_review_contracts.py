from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.sers.validation_routing_contracts import SERSValidationRouteKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSRouteScientificVerdict = Literal[
    "not_reviewable",
    "supportive",
    "inconsistent",
    "mixed",
    "indeterminate",
]

SERSEvidenceStrength = Literal[
    "none",
    "single_independent_source",
    "multiple_independent_sources",
    "experimental_result",
    "mixed_source_types",
]

SERSPreExperimentMechanismVerdict = Literal[
    "not_claimed",
    "supportive",
    "inconsistent",
    "mixed",
    "incomplete",
    "indeterminate",
]

SERSOutcomeScientificVerdict = Literal[
    "untested",
    "supportive",
    "inconsistent",
    "mixed",
    "indeterminate",
    "incomplete",
]

SERSExperimentReviewState = Literal[
    "not_candidate",
    "review_candidate",
    "direct_test_review_candidate",
    "experimental_result_available",
]

SERSResearchDecision = Literal[
    "resolve_route_interpretation",
    "collect_missing_route_evidence",
    "strengthen_route_evidence",
    "revise_component_claim",
    "resolve_conflicting_component_evidence",
    "consider_integrated_experiment_review",
    "candidate_for_integrated_experiment_review",
    "review_integrated_experiment_outcome",
    "outcome_inconsistent_reassess_mechanism",
    "provisional_routed_support",
    "preserve_indeterminate_state",
]

SERSClaimScopedFeedbackKind = Literal[
    "collect_evidence",
    "strengthen_evidence",
    "revise_component_claim",
    "resolve_evidence_conflict",
    "preserve_uncertainty",
    "review_experiment_requirement",
    "reassess_mechanism_after_outcome",
]


class SERSRouteScientificAssessment(StrictModel):
    assessment_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    route_kind: SERSValidationRouteKind

    required_for_mechanism_assessment: bool = False
    required_for_outcome_assessment: bool = False

    route_state: str = Field(min_length=1)
    verdict: SERSRouteScientificVerdict
    evidence_strength: SERSEvidenceStrength

    evidence_ids: list[str] = Field(default_factory=list)
    decisive_evidence_ids: list[str] = Field(default_factory=list)
    independent_source_count: int = 0
    experimental_result_count: int = 0
    blockers: list[str] = Field(default_factory=list)
    interpretation: str = Field(min_length=1)

    scientific_verdict_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False

    @model_validator(mode="after")
    def _consistency(self) -> "SERSRouteScientificAssessment":
        for values, label in (
            (self.evidence_ids, "evidence_ids"),
            (self.decisive_evidence_ids, "decisive_evidence_ids"),
            (self.blockers, "blockers"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if self.independent_source_count < 0 or self.experimental_result_count < 0:
            raise ValueError("source/result counts must be non-negative")
        return self


class SERSClaimScopedResearchFeedback(StrictModel):
    feedback_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    route_id: str | None = None
    route_kind: SERSValidationRouteKind | None = None
    feedback_kind: SERSClaimScopedFeedbackKind
    action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    applies_to_whole_hypothesis: Literal[False] = False
    canonical_feedback_permitted: Literal[False] = False
    automatic_hypothesis_rewrite_permitted: Literal[False] = False
    automatic_hypothesis_rejection_permitted: Literal[False] = False


class SERSHypothesisScientificReview(StrictModel):
    review_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_validation_state_id: str = Field(min_length=1)

    route_assessments: list[SERSRouteScientificAssessment] = Field(default_factory=list)
    pre_experiment_mechanism_verdict: SERSPreExperimentMechanismVerdict
    outcome_verdict: SERSOutcomeScientificVerdict

    experiment_review_state: SERSExperimentReviewState
    research_decision: SERSResearchDecision
    recommended_next_route_kind: SERSValidationRouteKind | None = None
    recommended_next_route_id: str | None = None
    recommended_next_action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    feedback: list[SERSClaimScopedResearchFeedback] = Field(default_factory=list)

    provisional_scientific_review_only: Literal[True] = True
    scientific_verdict_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    experiment_execution_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _consistency(self) -> "SERSHypothesisScientificReview":
        route_ids = [row.route_id for row in self.route_assessments]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route assessment route_id values must be unique")
        feedback_ids = [row.feedback_id for row in self.feedback]
        if len(feedback_ids) != len(set(feedback_ids)):
            raise ValueError("feedback_id values must be unique")
        if (self.recommended_next_route_kind is None) != (
            self.recommended_next_route_id is None
        ):
            raise ValueError(
                "recommended_next_route_kind and recommended_next_route_id must be set together"
            )
        return self


class SERSValidationReviewBundle(StrictModel):
    schema_version: Literal["sers-validation-review-bundle-v0"] = (
        "sers-validation-review-bundle-v0"
    )
    bundle_id: str = Field(min_length=1)
    source_portfolio_id: str = Field(min_length=1)
    source_validation_plan_bundle_id: str = Field(min_length=1)
    source_orchestration_bundle_id: str = Field(min_length=1)
    source_route_evidence_bundle_id: str | None = None

    reviews: list[SERSHypothesisScientificReview] = Field(default_factory=list)
    review_count: int

    shadow_only: Literal[True] = True
    provisional_scientific_review_only: Literal[True] = True
    scientific_verdict_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    experiment_promotion_authority: Literal[False] = False
    experiment_execution_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSValidationReviewBundle":
        if self.review_count != len(self.reviews):
            raise ValueError("review_count does not match reviews")
        ids = [row.review_id for row in self.reviews]
        if len(ids) != len(set(ids)):
            raise ValueError("review_id values must be unique")
        return self
