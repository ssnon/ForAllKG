from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.external_novelty_contracts import ExternalNoveltyStatus
from dac_her.hypothesis_contracts import HypothesisPortfolio


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


GapAction = Literal[
    "keep",
    "targeted_search_then_refine",
    "targeted_search_only",
    "refine_away_from_conflict",
    "reject",
]
RefinementDecision = Literal[
    "kept_original",
    "accepted_refinement",
    "abstained",
    "compile_rejected",
    "validation_rejected",
    "grounding_drift_rejected",
    "axis_fidelity_rejected",
    "internal_novelty_rejected",
    "external_novelty_rejected",
    "search_insufficient",
]


class NoveltyGap(StrictModel):
    gap_id: str
    hypothesis_id: str
    source_external_status: ExternalNoveltyStatus
    action: GapAction
    target_claim_ids: list[str] = Field(default_factory=list)
    differentiator: str
    already_known_boundary: list[str] = Field(default_factory=list)
    unresolved_boundary: list[str] = Field(default_factory=list)
    contextual_conflict_work_ids: list[str] = Field(default_factory=list)
    targeted_queries: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class NoveltyGapPlan(StrictModel):
    schema_version: Literal["novelty-gap-plan-v1"] = "novelty-gap-plan-v1"
    plan_id: str
    plan_sha256: str
    source_portfolio_id: str
    source_external_report_id: str
    gaps: list[NoveltyGap] = Field(default_factory=list)
    policy_version: Literal["novelty-gap-policy-v1"] = "novelty-gap-policy-v1"


class TargetedSearchRecord(StrictModel):
    hypothesis_id: str
    gap_id: str
    query_plan_id: str
    prior_art_packet_id: str
    external_report_id: str
    external_status_after_search: ExternalNoveltyStatus
    unique_work_count: int
    abstract_work_count: int
    successful_query_count: int


class RefinementAttempt(StrictModel):
    original_hypothesis_id: str
    final_hypothesis_id: str | None = None
    gap_id: str
    action: GapAction
    decision: RefinementDecision
    original_external_status: ExternalNoveltyStatus
    targeted_external_status: ExternalNoveltyStatus | None = None
    final_external_status: ExternalNoveltyStatus | None = None
    axis_fidelity_status: str | None = None
    internal_novelty_status: str | None = None
    grounding_preserved: bool = False
    refinement_generated: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str


class NoveltyRefinementReport(StrictModel):
    schema_version: Literal["novelty-refinement-report-v1"] = (
        "novelty-refinement-report-v1"
    )
    report_id: str
    report_sha256: str
    source_portfolio_id: str
    source_external_report_id: str
    source_gap_plan_id: str
    final_portfolio_id: str
    attempts: list[RefinementAttempt] = Field(default_factory=list)
    targeted_searches: list[TargetedSearchRecord] = Field(default_factory=list)
    accepted_refinement_count: int = 0
    kept_original_count: int = 0
    rejected_count: int = 0
    max_refinements_per_hypothesis: Literal[1] = 1
    external_prior_art_can_be_positive_premise: Literal[False] = False
    policy_version: Literal["novelty-refinement-policy-v1"] = (
        "novelty-refinement-policy-v1"
    )


class NoveltyRefinementArtifact(StrictModel):
    schema_version: Literal["novelty-refinement-artifact-v1"] = (
        "novelty-refinement-artifact-v1"
    )
    portfolio: HypothesisPortfolio
    report: NoveltyRefinementReport
