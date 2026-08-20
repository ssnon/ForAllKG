from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AxisFidelityStatus = Literal["pass", "warning", "fail"]
InternalNoveltyStatus = Literal[
    "reconstructs_existing_corpus_claim",
    "reconstructs_existing_corpus_chain",
    "corpus_supported_extension",
    "new_combination_within_corpus",
    "corpus_distinct_candidate",
    "insufficient_internal_evidence",
]
AxisAttemptStage = Literal[
    "initial",
    "compile_repair",
    "fidelity_repair",
    "novelty_repair",
]
AxisAttemptDecision = Literal[
    "accepted",
    "abstained",
    "compile_rejected",
    "validation_rejected",
    "fidelity_rejected",
    "novelty_rejected",
]


class DiscoveryAxis(StrictModel):
    axis_id: str
    axis_rank: int
    inspiration_id: str
    source_path_id: str
    candidate_unit_id: str
    label: str
    entry_anchor_id: str = ""
    entry_anchor_label: str = ""
    exit_anchor_id: str = ""
    exit_anchor_label: str = ""
    proposed_subject: str = ""
    proposed_relation: str = ""
    proposed_object: str = ""
    rendered_path: str
    source_mode: str
    exploration_score: float
    candidate_unit_score: float = 0.0
    planner_score: float
    mechanistic_continuity_band: str
    generic_entity_fraction: float = 0.0
    registry_hop_fraction: float = 0.0
    grounding_semantic_overlap: float = 0.0
    reaction_domain_switch_penalty: float = 0.0
    requires_verification: bool = True
    reason_codes: list[str] = Field(default_factory=list)


class DiscoveryAxisPlannerPolicy(StrictModel):
    policy_version: Literal["discovery-axis-planner-v1"] = "discovery-axis-planner-v1"
    max_axes: int = 5
    require_candidate_unit: bool = True
    min_exploration_score: float = 0.05
    min_candidate_unit_score: float = 0.30
    max_reaction_domain_switch_penalty: float = 0.50


class DiscoveryAxisPlan(StrictModel):
    schema_version: Literal["discovery-axis-plan-v1"] = "discovery-axis-plan-v1"
    plan_id: str
    plan_sha256: str
    source_dual_context_id: str
    source_dual_context_sha256: str
    source_bundle_id: str
    source_bundle_sha256: str
    corpus_id: str
    axes: list[DiscoveryAxis] = Field(default_factory=list)
    excluded_inspiration_ids: list[str] = Field(default_factory=list)
    policy: DiscoveryAxisPlannerPolicy


class AxisFidelityReview(StrictModel):
    schema_version: Literal["axis-fidelity-review-v1"] = "axis-fidelity-review-v1"
    axis_id: str
    hypothesis_id: str
    status: AxisFidelityStatus
    axis_signature_tokens: list[str] = Field(default_factory=list)
    matched_signature_tokens: list[str] = Field(default_factory=list)
    signature_coverage: float = 0.0
    hypothesis_similarity: float = 0.0
    bridge_similarity: float = 0.0
    prediction_similarity: float = 0.0
    combined_similarity: float = 0.0
    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str


class AxisAttemptRecord(StrictModel):
    axis_id: str
    stage: AxisAttemptStage
    generation_index: int
    decision: AxisAttemptDecision
    hypothesis_id: str | None = None
    title: str | None = None
    fidelity_status: AxisFidelityStatus | None = None
    internal_novelty_status: InternalNoveltyStatus | None = None
    compile_issue_codes: list[str] = Field(default_factory=list)
    validation_issue_codes: list[str] = Field(default_factory=list)
    repair_reason: str | None = None


class DiscoveryHypothesisLineage(StrictModel):
    hypothesis_id: str
    axis_id: str
    inspiration_id: str
    candidate_unit_id: str
    discovery_dependency: Literal["essential"] = "essential"
    epistemic_status: Literal["inspiration_only"] = "inspiration_only"
    axis_fidelity_status: AxisFidelityStatus
    internal_novelty_status: InternalNoveltyStatus
    fidelity_repaired: bool = False
    novelty_repaired: bool = False


class DiscoveryAxisSynthesisReport(StrictModel):
    schema_version: Literal["discovery-axis-synthesis-report-v1"] = (
        "discovery-axis-synthesis-report-v1"
    )
    report_id: str
    report_sha256: str
    source_dual_context_id: str
    source_dual_context_sha256: str
    axis_plan_id: str
    axis_plan_sha256: str
    final_portfolio_id: str
    final_portfolio_sha256: str
    attempted_axis_count: int
    accepted_hypothesis_count: int
    lineages: list[DiscoveryHypothesisLineage] = Field(default_factory=list)
    attempts: list[AxisAttemptRecord] = Field(default_factory=list)
    external_novelty_status: Literal["not_assessed"] = "not_assessed"
    policy_version: Literal["discovery-axis-synthesis-policy-v1"] = (
        "discovery-axis-synthesis-policy-v1"
    )


class DiscoveryAxisSynthesisArtifact(StrictModel):
    """Convenience envelope for consumers that want portfolio + lineage together."""

    schema_version: Literal["discovery-axis-synthesis-artifact-v1"] = (
        "discovery-axis-synthesis-artifact-v1"
    )
    portfolio: HypothesisPortfolio
    report: DiscoveryAxisSynthesisReport
