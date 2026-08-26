from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtStatus,
    ExternalNoveltyStatus,
    NoveltyClaimImportance,
    NoveltyClaimKind,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DistinctivenessEvidencePattern = Literal[
    "NO_REVIEWABLE_CORE_CLAIMS",
    "DIRECT_PRIOR_ART_SATURATED",
    "RELATION_BACKED_SATURATED",
    "HIGHER_ORDER_RELATIONAL_GAP_WITH_LOWER_ORDER_PRIOR_ART",
    "LOWER_ORDER_PRIOR_ART_PRESENT",
    "SEARCH_BOUNDED_UNMATCHED",
    "SEARCH_COVERAGE_LIMITED",
    "MIXED_PRIOR_ART",
]


SemanticDistinctivenessStatus = Literal[
    "UNASSESSED",
]


class ScientificDistinctivenessSemanticDimensions(StrictModel):
    """Semantic non-obviousness dimensions intentionally deferred in v1.

    The existing external-novelty artifacts establish search-bounded
    prior-art structure, but they do not by themselves justify judgments
    such as "straightforward conjunction" or "mechanism switch".

    v1 therefore records these dimensions explicitly as unassessed rather
    than silently inventing deterministic heuristics for them.
    """

    conceptual_prior_art_density: SemanticDistinctivenessStatus = (
        "UNASSESSED"
    )
    straightforward_conjunction: SemanticDistinctivenessStatus = (
        "UNASSESSED"
    )
    mechanism_switch: SemanticDistinctivenessStatus = (
        "UNASSESSED"
    )
    ranking_or_regime_change: SemanticDistinctivenessStatus = (
        "UNASSESSED"
    )
    counterfactual_distinctiveness: SemanticDistinctivenessStatus = (
        "UNASSESSED"
    )
    evidence_family_complementarity: SemanticDistinctivenessStatus = (
        "UNASSESSED"
    )


class ScientificDistinctivenessClaimSignal(StrictModel):
    hypothesis_id: str
    claim_id: str
    claim_kind: NoveltyClaimKind
    importance: NoveltyClaimImportance
    claim_text: str

    prior_art_status: ClaimPriorArtStatus

    query_count: int = Field(ge=0)
    successful_query_count: int = Field(ge=0)
    unique_work_count: int = Field(ge=0)
    abstract_work_count: int = Field(ge=0)
    reviewed_work_count: int = Field(ge=0)

    relationship_counts: dict[str, int] = Field(
        default_factory=dict
    )

    direct_prior_art_work_ids: list[str] = Field(
        default_factory=list
    )
    partial_prior_art_work_ids: list[str] = Field(
        default_factory=list
    )
    lower_order_prior_art_work_ids: list[str] = Field(
        default_factory=list
    )
    directional_counterevidence_work_ids: list[str] = Field(
        default_factory=list
    )
    contextual_conflict_work_ids: list[str] = Field(
        default_factory=list
    )
    conflicting_prior_art_work_ids: list[str] = Field(
        default_factory=list
    )

    reason_codes: list[str] = Field(
        default_factory=list
    )


class ScientificDistinctivenessReview(StrictModel):
    hypothesis_id: str
    title: str

    external_novelty_status: ExternalNoveltyStatus
    evidence_pattern: DistinctivenessEvidencePattern

    claim_count: int = Field(ge=0)
    core_claim_count: int = Field(ge=0)

    direct_prior_art_core_claim_count: int = Field(ge=0)
    relation_backed_core_claim_count: int = Field(ge=0)
    component_supported_core_claim_count: int = Field(ge=0)
    no_direct_match_core_claim_count: int = Field(ge=0)
    lower_order_supported_core_claim_count: int = Field(ge=0)

    direct_prior_art_core_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    relation_backed_core_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    component_supported_core_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    no_direct_match_core_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )
    lower_order_supported_core_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )

    higher_order_relational_gap_claim_count: int = Field(
        ge=0
    )
    lower_order_core_unique_work_count: int = Field(
        ge=0
    )
    directional_counterevidence_unique_work_count: int = Field(
        ge=0
    )

    search_coverage_sufficient: bool
    search_unique_work_count: int = Field(ge=0)
    search_abstract_work_count: int = Field(ge=0)

    claim_signals: list[
        ScientificDistinctivenessClaimSignal
    ] = Field(default_factory=list)

    semantic_dimensions: (
        ScientificDistinctivenessSemanticDimensions
    ) = Field(
        default_factory=(
            ScientificDistinctivenessSemanticDimensions
        )
    )

    source_claim_ids: list[str] = Field(
        default_factory=list
    )
    referenced_prior_art_work_ids: list[str] = Field(
        default_factory=list
    )

    source_aggregate_warnings: list[str] = Field(
        default_factory=list
    )

    reason_codes: list[str] = Field(
        default_factory=list
    )
    interpretation: str

    diagnostic_only: Literal[True] = True
    retrieval_performed: Literal[False] = False
    model_review_performed: Literal[False] = False
    action_policy_applied: Literal[False] = False
    scientific_selection_changed: Literal[False] = False
    positive_premise_surface_changed: Literal[False] = False


class ScientificDistinctivenessReport(StrictModel):
    schema_version: Literal[
        "scientific-distinctiveness-report-v1"
    ] = "scientific-distinctiveness-report-v1"

    report_id: str
    report_sha256: str

    source_portfolio_id: str

    source_external_novelty_report_id: str
    source_external_novelty_report_sha256: str

    source_query_plan_id: str
    source_query_plan_sha256: str

    source_prior_art_packet_id: str
    source_prior_art_packet_sha256: str

    source_searched_at_utc: str

    reviews: list[
        ScientificDistinctivenessReview
    ] = Field(default_factory=list)

    evidence_pattern_counts: dict[str, int] = Field(
        default_factory=dict
    )

    source_aggregate_warning_count: int = Field(ge=0)

    diagnostic_scope: Literal[
        "existing_external_prior_art_evidence_only"
    ] = "existing_external_prior_art_evidence_only"

    semantic_dimensions_assessed: Literal[False] = False

    retrieval_performed: Literal[False] = False
    model_review_performed: Literal[False] = False
    action_policy_applied: Literal[False] = False
    scientific_selection_changed: Literal[False] = False

    epistemic_usage: Literal[
        "diagnostic_only_existing_prior_art_not_positive_premise"
    ] = (
        "diagnostic_only_existing_prior_art_not_positive_premise"
    )
