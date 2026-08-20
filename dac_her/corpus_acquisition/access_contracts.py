from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AccessStatus = Literal[
    "resolved_direct_pdf",
    "resolved_landing_only",
    "unresolved",
]

ResolverStatus = Literal[
    "success",
    "skipped",
    "failed",
]

ArtifactStatus = Literal[
    "downloaded",
    "not_attempted",
    "download_failed",
]


class ResolverAttempt(StrictModel):
    resolver: str
    status: ResolverStatus
    elapsed_seconds: float = 0.0
    message: str | None = None


class AccessLocation(StrictModel):
    location_id: str
    resolver: str
    url: str
    url_for_pdf: str | None = None
    url_for_landing_page: str | None = None
    is_oa: bool = True
    is_best: bool = False
    host_type: str | None = None
    version: str | None = None
    license: str | None = None
    source_id: str | None = None
    source_name: str | None = None
    automatic_download_eligible: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class AccessResolution(StrictModel):
    schema_version: Literal["access-resolution-v1"] = "access-resolution-v1"
    work_id: str
    doi: str | None = None
    status: AccessStatus
    locations: list[AccessLocation] = Field(default_factory=list)
    resolver_attempts: list[ResolverAttempt] = Field(default_factory=list)
    selected_location_id: str | None = None
    selected_download_url: str | None = None
    resolution_notes: list[str] = Field(default_factory=list)
    paywall_bypass_attempted: Literal[False] = False


class ArtifactDownloadAttempt(StrictModel):
    location_id: str
    url: str
    host: str | None = None
    status: Literal["success", "failed"]
    elapsed_seconds: float = 0.0
    resolved_url: str | None = None
    content_type: str | None = None
    byte_count: int | None = None
    error_code: str | None = None
    error: str | None = None


class SourceArtifact(StrictModel):
    schema_version: Literal["source-artifact-v1"] = "source-artifact-v1"
    artifact_id: str
    work_id: str
    role: Literal[
        "main",
        "supporting_information",
        "dataset",
        "supplementary_media",
    ]
    status: ArtifactStatus
    source_url: str | None = None
    resolved_url: str | None = None
    local_path: str | None = None
    sha256: str | None = None
    byte_count: int | None = None
    content_type: str | None = None
    license: str | None = None
    version: str | None = None
    host_type: str | None = None
    acquired_at_utc: str | None = None
    acquisition_method: str | None = None
    error: str | None = None
    selected_location_id: str | None = None
    attempted_location_count: int = 0
    download_attempts: list[ArtifactDownloadAttempt] = Field(
        default_factory=list
    )
    positive_evidence_promotion_performed: Literal[False] = False


class SourceAcquisitionPolicy(StrictModel):
    schema_version: Literal[
        "source-acquisition-policy-v1"
    ] = "source-acquisition-policy-v1"
    policy_id: str
    unpaywall_email_env: str = "UNPAYWALL_EMAIL"
    fallback_email_env: str = "CROSSREF_MAILTO"
    openalex_api_key_env: str = "OPENALEX_API_KEY"
    openalex_mailto_env: str = "OPENALEX_MAILTO"
    openalex_require_api_key: bool = True
    use_unpaywall: bool = True
    use_openalex: bool = True
    use_pmc_aws: bool = False
    use_catalog_open_access_url: bool = True
    request_timeout_seconds: float = Field(default=45.0, gt=0)
    retries: int = Field(default=2, ge=0, le=6)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    resolver_delay_seconds: float = Field(default=0.10, ge=0)
    max_artifact_bytes: int = Field(
        default=104857600,
        ge=1048576,
    )
    require_pdf_magic: bool = True
    user_agent: str = "GraphAgentsDAC-CorpusAcquisition/M3"
    download_user_agent: str = (
        "Mozilla/5.0 (compatible; GraphAgentsDAC-CorpusAcquisition/M3)"
    )
    try_all_direct_pdf_locations: bool = True
    send_landing_page_referer: bool = True
    auto_download_main: bool = True
    allow_catalog_oa_fallback: bool = True
    supplementary_discovery: Literal[
        "deferred_to_m3_1"
    ] = "deferred_to_m3_1"


class CorpusSourceAcquisitionReport(StrictModel):
    schema_version: Literal[
        "corpus-source-acquisition-report-v1"
    ] = "corpus-source-acquisition-report-v1"
    acquisition_id: str
    source_profile_id: str
    source_catalog_id: str
    source_selection_report_path: str
    policy_id: str
    selected_work_count: int
    access_resolved_direct_pdf_count: int = 0
    access_resolved_landing_only_count: int = 0
    access_unresolved_count: int = 0
    artifact_downloaded_count: int = 0
    artifact_download_failed_count: int = 0
    artifact_not_attempted_count: int = 0
    unpaywall_attempt_count: int = 0
    unpaywall_success_count: int = 0
    openalex_attempt_count: int = 0
    openalex_success_count: int = 0
    openalex_skipped_count: int = 0
    openalex_location_count: int = 0
    openalex_direct_pdf_work_count: int = 0
    openalex_incremental_direct_pdf_work_count: int = 0
    openalex_artifact_download_count: int = 0
    catalog_oa_fallback_count: int = 0
    total_download_location_attempts: int = 0
    multi_location_recovery_count: int = 0
    download_failure_reason_counts: dict[str, int] = Field(default_factory=dict)
    download_host_stats: dict[str, dict[str, int]] = Field(default_factory=dict)
    upstream_provider_query_success_count: int | None = None
    upstream_provider_query_execution_count: int | None = None
    upstream_coverage_warning: bool = False
    output_root: str
    supplementary_discovery: Literal[
        "deferred_to_m3_1"
    ] = "deferred_to_m3_1"
    paywall_bypass_attempted: Literal[False] = False
    positive_evidence_promotion_performed: Literal[False] = False
