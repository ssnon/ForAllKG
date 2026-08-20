from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


QualityStatus = Literal["pass", "manual_review", "exclude"]


class CorpusQualityPolicy(StrictModel):
    schema_version: Literal[
        "corpus-quality-policy-v1"
    ] = "corpus-quality-policy-v1"
    policy_id: str

    hard_exclude_title_patterns: list[str] = Field(default_factory=list)
    manual_review_title_patterns: list[str] = Field(default_factory=list)
    hard_exclude_publication_types: list[str] = Field(default_factory=list)
    manual_review_publication_types: list[str] = Field(default_factory=list)
    manual_review_doi_prefixes: list[str] = Field(default_factory=list)
    manual_review_venue_terms: list[str] = Field(default_factory=list)

    primary_topic_terms: list[str] = Field(default_factory=list)
    title_context_terms: list[str] = Field(default_factory=list)
    min_title_context_matches_without_primary_topic: int = Field(
        default=1,
        ge=0,
    )
    require_primary_topic_signal: bool = True
    require_title_grounding: bool = True

    allow_manual_review_for_auto_selection: bool = False

    @model_validator(mode="after")
    def _validate_patterns(self) -> "CorpusQualityPolicy":
        import re

        for pattern in [
            *self.hard_exclude_title_patterns,
            *self.manual_review_title_patterns,
        ]:
            re.compile(pattern, re.I)
        return self


class CorpusQualityAssessment(StrictModel):
    work_id: str
    title: str
    doi: str | None = None
    status: QualityStatus
    reasons: list[str] = Field(default_factory=list)
    matched_primary_topic_terms_title: list[str] = Field(default_factory=list)
    matched_primary_topic_terms_abstract: list[str] = Field(default_factory=list)
    matched_title_context_terms: list[str] = Field(default_factory=list)
    original_m2_eligibility_status: str
    originally_selected: bool = False


class CorpusQualityGateReport(StrictModel):
    schema_version: Literal[
        "corpus-quality-gate-report-v1"
    ] = "corpus-quality-gate-report-v1"
    quality_gate_id: str
    policy_id: str
    profile_id: str
    source_catalog_id: str
    candidate_count: int
    upstream_eligible_count: int
    quality_pass_count: int
    quality_manual_review_count: int
    quality_exclude_count: int
    original_selected_count: int
    retained_original_selected_count: int
    dropped_original_selected_count: int
    replacement_selected_count: int
    final_selected_count: int
    target_total: int
    reason_counts: dict[str, int] = Field(default_factory=dict)
    final_unfilled_axis_quotas: dict[str, int] = Field(default_factory=dict)
    final_selected_work_ids: list[str] = Field(default_factory=list)
    positive_evidence_promotion_performed: Literal[False] = False
