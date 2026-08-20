from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopePolicy(StrictModel):
    required_term_groups: list[list[str]] = Field(default_factory=list)
    excluded_title_terms: list[str] = Field(default_factory=list)
    excluded_publication_types: list[str] = Field(default_factory=list)
    min_year: int | None = None
    max_year: int | None = None
    require_abstract: bool = False
    manual_review_if_no_abstract: bool = False
    require_axis_match: bool = True

    @model_validator(mode="after")
    def _nonempty_required_groups(self) -> "ScopePolicy":
        for group in self.required_term_groups:
            if not group:
                raise ValueError(
                    "required_term_groups cannot contain an empty group"
                )
        if (
            self.min_year is not None
            and self.max_year is not None
            and self.min_year > self.max_year
        ):
            raise ValueError("min_year must be <= max_year")
        return self


class AcquisitionAxis(StrictModel):
    axis_id: str
    target_selected: int = Field(ge=0)
    queries: list[str] = Field(min_length=1)
    indicators: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0)


class SignalRule(StrictModel):
    signal_id: str
    terms: list[str] = Field(min_length=1)
    weight: float
    match_mode: Literal["any", "all"] = "any"


class CitationBonusRule(StrictModel):
    min_citations: int = Field(ge=0)
    bonus: float


class ScorePolicy(StrictModel):
    open_access_bonus: float = 0.0
    abstract_available_bonus: float = 0.0
    retrieval_axis_bonus: float = 0.0
    axis_indicator_bonus: float = 0.0
    max_axis_bonus: float | None = Field(default=None, ge=0.0)
    signals: list[SignalRule] = Field(default_factory=list)
    citation_bonuses: list[CitationBonusRule] = Field(default_factory=list)


class SelectionPolicy(StrictModel):
    target_total: int = Field(gt=0)
    include_manual_review: bool = False

    @model_validator(mode="after")
    def _positive_target(self) -> "SelectionPolicy":
        if self.target_total <= 0:
            raise ValueError("target_total must be positive")
        return self


class DiscoveryPolicy(StrictModel):
    results_per_query: int = Field(default=50, ge=1, le=100)
    default_providers: list[
        Literal["semantic_scholar", "crossref"]
    ] = Field(
        default_factory=lambda: [
            "semantic_scholar",
            "crossref",
        ]
    )


class AcquisitionProfile(StrictModel):
    schema_version: Literal[
        "corpus-acquisition-profile-v1"
    ] = "corpus-acquisition-profile-v1"
    profile_id: str
    domain_profile_id: str
    description: str = ""
    discovery: DiscoveryPolicy = Field(
        default_factory=DiscoveryPolicy
    )
    scope: ScopePolicy = Field(
        default_factory=ScopePolicy
    )
    scoring: ScorePolicy = Field(
        default_factory=ScorePolicy
    )
    selection: SelectionPolicy
    axes: list[AcquisitionAxis] = Field(min_length=1)

    @model_validator(mode="after")
    def _profile_invariants(self) -> "AcquisitionProfile":
        axis_ids = [axis.axis_id for axis in self.axes]
        if len(axis_ids) != len(set(axis_ids)):
            raise ValueError("duplicate axis_id")
        signal_ids = [
            signal.signal_id
            for signal in self.scoring.signals
        ]
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("duplicate signal_id")
        quota_sum = sum(
            axis.target_selected
            for axis in self.axes
        )
        if quota_sum > self.selection.target_total:
            raise ValueError(
                "sum(axis.target_selected) must be <= selection.target_total; "
                "each selected work receives at most one primary quota axis"
            )
        return self


CandidateEligibility = Literal[
    "eligible",
    "excluded",
    "manual_review",
]


class CandidateAssessment(StrictModel):
    work_id: str
    title: str
    doi: str | None = None
    year: int | None = None
    eligibility_status: CandidateEligibility
    exclusion_reasons: list[str] = Field(default_factory=list)
    matched_axes: list[str] = Field(default_factory=list)
    matched_terms_by_axis: dict[str, list[str]] = Field(default_factory=dict)
    score_components: dict[str, float] = Field(default_factory=dict)
    total_score: float = 0.0
    open_access_available: bool = False
    abstract_available: bool = False
    scientific_result_direction_inferred: Literal[False] = False


class SelectedCorpusWork(StrictModel):
    work_id: str
    title: str
    doi: str | None = None
    year: int | None = None
    venue: str | None = None
    open_access_url: str | None = None
    matched_axes: list[str] = Field(default_factory=list)
    primary_quota_axis: str | None = None
    total_score: float
    acquisition_status: Literal[
        "selected_metadata_only"
    ] = "selected_metadata_only"


class CorpusSelectionReport(StrictModel):
    schema_version: Literal[
        "corpus-selection-report-v1"
    ] = "corpus-selection-report-v1"
    profile_id: str
    source_catalog_id: str
    candidate_count: int
    eligible_count: int
    manual_review_count: int
    excluded_count: int
    selected_count: int
    target_total: int
    axis_candidate_counts: dict[str, int] = Field(default_factory=dict)
    axis_quota_targets: dict[str, int] = Field(default_factory=dict)
    axis_primary_selected_counts: dict[str, int] = Field(default_factory=dict)
    unfilled_axis_quotas: dict[str, int] = Field(default_factory=dict)
    selected_work_ids: list[str] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)
    positive_evidence_promotion_performed: Literal[False] = False
