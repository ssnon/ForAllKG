from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_contracts import (
    IdeaTransitionAssessment,
    ResearchIdeaNode,
    ResearchIdeaSearchState,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


IdeaSearchAction = Literal[
    "VERIFY",
    "ALTERNATE_REALIZATION",
    "SAME_PREMISE_SHARPEN",
    "EVIDENCE_REAXIS",
    "AXIS_MUTATION",
    "REQUEST_GRAPH_RETRAVERSAL",
    "REFRAME",
]

IdeaPriorityBand = Literal[
    "HIGH",
    "MEDIUM",
    "LOW",
    "DEFER",
]

IdeaObservationScope = Literal[
    "REALIZATION",
    "IDEA",
    "FAMILY",
]


class IdeaOutcomeObservation(StrictModel):
    """Policy-free observation projected from one grounded realization.

    This artifact records what happened to a realization. It deliberately does
    not decide whether the parent research idea survives, reproduces, or gets
    more compute.
    """

    schema_version: Literal[
        "idea-outcome-observation-v1"
    ] = "idea-outcome-observation-v1"

    observation_id: str = Field(min_length=1)
    idea_id: str = Field(min_length=1)
    target_scope: IdeaObservationScope = "REALIZATION"
    source_systems: list[str] = Field(default_factory=list)
    source_versions: list[str] = Field(default_factory=list)
    candidate_id: str | None = None
    hypothesis_id: str | None = None

    materialization_status: str | None = None
    materialization_issue_codes: list[str] = Field(default_factory=list)

    residual_epistemic_state: str | None = None
    residual_state_reason: str | None = None
    external_status: str | None = None

    prospective_identifiability: str | None = None
    current_evidence_status: str | None = None
    directionality_mode: str | None = None
    measurement_compatibility_mode: str | None = None
    prospective_contract_integrity_passed: bool | None = None

    source_report_ids: list[str] = Field(default_factory=list)

    observation_only: Literal[True] = True
    search_policy_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class IdeaFeedbackCoverageAudit(StrictModel):
    schema_version: Literal[
        "idea-feedback-coverage-audit-v1"
    ] = "idea-feedback-coverage-audit-v1"

    materialized_hypothesis_ids: list[str] = Field(default_factory=list)
    materialized_hypothesis_count: int = Field(ge=0, default=0)
    residual_matched_hypothesis_ids: list[str] = Field(default_factory=list)
    residual_matched_hypothesis_count: int = Field(ge=0, default=0)
    prospective_matched_hypothesis_ids: list[str] = Field(default_factory=list)
    prospective_matched_hypothesis_count: int = Field(ge=0, default=0)
    residual_unmatched_hypothesis_ids: list[str] = Field(default_factory=list)
    prospective_unmatched_hypothesis_ids: list[str] = Field(default_factory=list)
    residual_coverage_fraction: float = Field(ge=0.0, le=1.0, default=0.0)
    prospective_coverage_fraction: float = Field(ge=0.0, le=1.0, default=0.0)
    any_post_verification_feedback: bool = False
    exact_lineage_required: Literal[True] = True

    @model_validator(mode="after")
    def _validate_feedback_counts(self) -> "IdeaFeedbackCoverageAudit":
        self.materialized_hypothesis_ids = list(
            dict.fromkeys(self.materialized_hypothesis_ids)
        )
        self.residual_matched_hypothesis_ids = list(
            dict.fromkeys(self.residual_matched_hypothesis_ids)
        )
        self.prospective_matched_hypothesis_ids = list(
            dict.fromkeys(self.prospective_matched_hypothesis_ids)
        )
        if self.materialized_hypothesis_count != len(self.materialized_hypothesis_ids):
            raise ValueError("materialized_hypothesis_count mismatch")
        if self.residual_matched_hypothesis_count != len(
            self.residual_matched_hypothesis_ids
        ):
            raise ValueError("residual_matched_hypothesis_count mismatch")
        if self.prospective_matched_hypothesis_count != len(
            self.prospective_matched_hypothesis_ids
        ):
            raise ValueError("prospective_matched_hypothesis_count mismatch")
        return self


class IdeaSearchDirective(StrictModel):
    """Versioned search-policy interpretation of one or more observations."""

    schema_version: Literal[
        "idea-search-directive-v1"
    ] = "idea-search-directive-v1"

    idea_id: str = Field(min_length=1)
    observation_ids: list[str] = Field(default_factory=list)

    direct_realization_reproductive: bool
    idea_reproductive: bool
    priority_band: IdeaPriorityBand
    preferred_actions: list[IdeaSearchAction] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    policy_version: Literal[
        "idea-feedback-policy-v1"
    ] = "idea-feedback-policy-v1"

    shadow_search_policy_authority: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_policy_shape(self) -> "IdeaSearchDirective":
        if not self.idea_reproductive and self.direct_realization_reproductive:
            raise ValueError(
                "direct realization reproduction requires idea_reproductive=true"
            )
        self.observation_ids = list(dict.fromkeys(self.observation_ids))
        self.preferred_actions = list(dict.fromkeys(self.preferred_actions))
        self.reason_codes = list(dict.fromkeys(self.reason_codes))
        return self


class IdeaSearchCandidateState(StrictModel):
    idea_id: str = Field(min_length=1)
    candidate_ids: list[str] = Field(default_factory=list)
    eligible_profiles: list[str] = Field(default_factory=list)
    pareto_layer_by_profile: dict[str, int] = Field(default_factory=dict)
    dimension_levels: dict[str, int] = Field(default_factory=dict)
    verification_burden: str = "MODERATE"
    legacy_portfolio_retained: bool = False
    legacy_family_signatures: list[str] = Field(default_factory=list)

    priority_band: IdeaPriorityBand = "MEDIUM"
    underexplored_bonus: float = Field(ge=0.0)
    visit_count: int = Field(ge=0)
    selected_for_g2: bool = False
    assigned_profile: str | None = None
    selection_reason_codes: list[str] = Field(default_factory=list)


class ExactKernelDuplicateGroup(StrictModel):
    kernel_sha256: str = Field(min_length=64, max_length=64)
    idea_ids: list[str] = Field(min_length=2)
    generation_indices: list[int] = Field(min_length=1)


class IdeaGenerationState(StrictModel):
    schema_version: Literal[
        "idea-generation-state-v1"
    ] = "idea-generation-state-v1"

    generation_index: int = Field(ge=0)
    population_idea_ids: list[str] = Field(default_factory=list)
    evaluated_idea_ids: list[str] = Field(default_factory=list)
    materialized_idea_ids: list[str] = Field(default_factory=list)
    observed_idea_ids: list[str] = Field(default_factory=list)
    selected_parent_idea_ids: list[str] = Field(default_factory=list)

    max_parent_budget: int = Field(ge=0)
    selected_parent_count: int = Field(ge=0)

    shadow_only: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "IdeaGenerationState":
        if self.selected_parent_count != len(self.selected_parent_idea_ids):
            raise ValueError("selected_parent_count mismatch")
        return self


class GenerationalIdeaSearchShadowReport(StrictModel):
    schema_version: Literal[
        "generational-idea-search-shadow-v1"
    ] = "generational-idea-search-shadow-v1"

    report_id: str
    report_sha256: str
    source_population_id: str
    source_evolution_report_id: str
    source_pool_id: str
    source_evaluation_report_id: str
    source_selection_id: str
    source_materialization_report_id: str

    research_ideas: list[ResearchIdeaNode] = Field(default_factory=list)
    research_idea_count: int = Field(ge=0)
    generation0_idea_count: int = Field(ge=0)
    generation1_idea_count: int = Field(ge=0)
    exact_kernel_unique_count: int = Field(ge=0)
    exact_kernel_duplicate_group_count: int = Field(ge=0)
    exact_kernel_duplicate_groups: list[ExactKernelDuplicateGroup] = Field(
        default_factory=list
    )

    transitions: list[IdeaTransitionAssessment] = Field(default_factory=list)
    transition_count: int = Field(ge=0)
    identity_relation_counts: dict[str, int] = Field(default_factory=dict)
    genealogy_relation_counts: dict[str, int] = Field(default_factory=dict)
    transition_counts_by_operator: dict[str, int] = Field(default_factory=dict)
    operator_expectation_inconsistent_count: int = Field(ge=0)
    semantic_noop_count: int = Field(ge=0)
    local_repair_boundary_cross_count: int = Field(ge=0)

    observations: list[IdeaOutcomeObservation] = Field(default_factory=list)
    observation_count: int = Field(ge=0)
    observation_count_by_materialization_status: dict[str, int] = Field(
        default_factory=dict
    )
    observation_count_by_residual_state: dict[str, int] = Field(
        default_factory=dict
    )
    observation_count_by_prospective_status: dict[str, int] = Field(
        default_factory=dict
    )
    feedback_coverage: IdeaFeedbackCoverageAudit = Field(
        default_factory=IdeaFeedbackCoverageAudit
    )
    directives: list[IdeaSearchDirective] = Field(default_factory=list)
    search_states: list[ResearchIdeaSearchState] = Field(default_factory=list)
    candidate_states: list[IdeaSearchCandidateState] = Field(default_factory=list)

    generation_states: list[IdeaGenerationState] = Field(min_length=3, max_length=3)
    g2_parent_idea_ids: list[str] = Field(default_factory=list)
    g2_parent_count: int = Field(ge=0)
    g2_parent_profile_counts: dict[str, int] = Field(default_factory=dict)
    legacy_portfolio_overlap_count: int = Field(ge=0)

    baseline_g2_parent_idea_ids: list[str] = Field(default_factory=list)
    baseline_g2_parent_count: int = Field(ge=0, default=0)
    baseline_g2_parent_profile_counts: dict[str, int] = Field(default_factory=dict)
    baseline_priority_band_counts: dict[str, int] = Field(default_factory=dict)
    feedback_priority_band_counts: dict[str, int] = Field(default_factory=dict)
    feedback_changed_parent_set: bool = False
    feedback_parent_added_idea_ids: list[str] = Field(default_factory=list)
    feedback_parent_dropped_idea_ids: list[str] = Field(default_factory=list)
    feedback_parent_overlap_count: int = Field(ge=0, default=0)

    feedback_observation_policy_separated: Literal[True] = True
    exact_kernel_duplicate_suppression_only: Literal[True] = True
    soft_family_not_used_as_hard_gate: Literal[True] = True
    single_scalar_search_score_used: Literal[False] = False
    existing_frontier_or_evolution_mutated: Literal[False] = False
    new_llm_calls: Literal[False] = False
    new_retrieval_calls: Literal[False] = False
    shadow_only: Literal[True] = True
    search_parent_selection_authority: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "GenerationalIdeaSearchShadowReport":
        if self.research_idea_count != len(self.research_ideas):
            raise ValueError("research_idea_count mismatch")
        if self.transition_count != len(self.transitions):
            raise ValueError("transition_count mismatch")
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count mismatch")
        if self.g2_parent_count != len(self.g2_parent_idea_ids):
            raise ValueError("g2_parent_count mismatch")
        if self.baseline_g2_parent_count != len(self.baseline_g2_parent_idea_ids):
            raise ValueError("baseline_g2_parent_count mismatch")
        if self.research_idea_count != (
            self.generation0_idea_count + self.generation1_idea_count
        ):
            raise ValueError("research idea generation counts mismatch")
        if self.transition_count != sum(self.identity_relation_counts.values()):
            raise ValueError("transition identity counts mismatch")
        return self


__all__ = [
    "ExactKernelDuplicateGroup",
    "GenerationalIdeaSearchShadowReport",
    "IdeaFeedbackCoverageAudit",
    "IdeaGenerationState",
    "IdeaOutcomeObservation",
    "IdeaObservationScope",
    "IdeaPriorityBand",
    "IdeaSearchAction",
    "IdeaSearchCandidateState",
    "IdeaSearchDirective",
]
