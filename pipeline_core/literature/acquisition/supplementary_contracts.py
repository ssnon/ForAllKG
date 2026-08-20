from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SupplementaryCandidateKind = Literal[
    "direct_file",
    "supplementary_landing",
    "related_identifier",
]

SupplementaryDiscoveryStatus = Literal[
    "direct_file_candidates",
    "metadata_only_candidates",
    "unresolved",
]


class SupplementaryResolverAttempt(StrictModel):
    resolver: str
    status: Literal["success", "skipped", "failed"]
    elapsed_seconds: float = 0.0
    result_count: int = 0
    message: str | None = None


class SupplementaryCandidate(StrictModel):
    candidate_id: str
    work_id: str
    kind: SupplementaryCandidateKind
    resolver: str
    source_page_url: str | None = None
    resolved_source_page_url: str | None = None
    url: str | None = None
    identifier: str | None = None
    identifier_type: str | None = None
    relation_type: str | None = None
    anchor_text: str | None = None
    title_hint: str | None = None
    content_type_hint: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    automatic_download_eligible: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class SupplementaryDiscovery(StrictModel):
    schema_version: Literal[
        "supplementary-discovery-v1"
    ] = "supplementary-discovery-v1"
    work_id: str
    doi: str | None = None
    status: SupplementaryDiscoveryStatus
    candidates: list[SupplementaryCandidate] = Field(default_factory=list)
    resolver_attempts: list[SupplementaryResolverAttempt] = Field(
        default_factory=list
    )
    scanned_landing_pages: list[str] = Field(default_factory=list)
    discovery_notes: list[str] = Field(default_factory=list)
    publisher_specific_url_guessing_performed: Literal[False] = False
    paywall_bypass_attempted: Literal[False] = False


class SupplementaryDiscoveryPolicy(StrictModel):
    schema_version: Literal[
        "supplementary-discovery-policy-v1"
    ] = "supplementary-discovery-policy-v1"
    policy_id: str
    crossref_mailto_env: str = "CROSSREF_MAILTO"
    use_crossref_relations: bool = True
    use_public_landing_html: bool = True
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    retries: int = Field(default=2, ge=0, le=6)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    resolver_delay_seconds: float = Field(default=0.25, ge=0)
    max_landing_pages_per_work: int = Field(default=3, ge=0, le=10)
    max_candidates_per_work: int = Field(default=12, ge=1, le=100)
    max_html_bytes: int = Field(default=5242880, ge=65536)
    max_artifact_bytes: int = Field(default=157286400, ge=1048576)
    auto_download_high_confidence_direct_files: bool = True
    allow_medium_confidence_direct_files: bool = False
    user_agent: str = "GraphAgentsDAC-CorpusAcquisition/M3.1"


class SupplementaryAcquisitionReport(StrictModel):
    schema_version: Literal[
        "supplementary-acquisition-report-v1"
    ] = "supplementary-acquisition-report-v1"
    acquisition_id: str
    source_profile_id: str
    source_catalog_id: str
    source_m3_report_path: str
    policy_id: str
    selected_work_count: int
    direct_file_candidate_work_count: int = 0
    metadata_only_candidate_work_count: int = 0
    unresolved_work_count: int = 0
    candidate_count: int = 0
    high_confidence_candidate_count: int = 0
    medium_confidence_candidate_count: int = 0
    low_confidence_candidate_count: int = 0
    supplementary_artifact_downloaded_count: int = 0
    supplementary_artifact_download_failed_count: int = 0
    supplementary_artifact_not_attempted_count: int = 0
    crossref_relation_attempt_count: int = 0
    public_landing_attempt_count: int = 0
    output_root: str
    publisher_specific_url_guessing_performed: Literal[False] = False
    paywall_bypass_attempted: Literal[False] = False
    positive_evidence_promotion_performed: Literal[False] = False
