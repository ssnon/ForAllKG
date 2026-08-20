from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AcquisitionAwareBackfillPolicy(StrictModel):
    schema_version: Literal[
        "acquisition-aware-backfill-policy-v1"
    ] = "acquisition-aware-backfill-policy-v1"
    policy_id: str

    required_quality_status: Literal["pass"] = "pass"
    preserve_axis_quotas: bool = True

    # OA metadata is never added to the scientific score. It is used only as
    # a deterministic tie-break after equal scientific total_score.
    oa_hint_tiebreak_only: bool = True

    reuse_existing_downloaded_main: bool = True
    treat_existing_non_downloaded_as_exhausted: bool = True
    global_fill_after_quota_attempts: bool = True

    max_new_candidate_attempts: int | None = Field(
        default=None,
        ge=1,
    )

    @model_validator(mode="after")
    def _invariants(self) -> "AcquisitionAwareBackfillPolicy":
        if self.required_quality_status != "pass":
            raise ValueError(
                "M3.2 automatic backfill is restricted to quality=pass"
            )
        return self


BackfillOutcome = Literal[
    "retained_existing_download",
    "downloaded",
    "download_failed",
    "not_attempted",
]


class BackfillAttempt(StrictModel):
    attempt_index: int = Field(ge=1)
    work_id: str
    title: str
    requested_axis: str | None = None
    phase: Literal["quota", "global_fill"]
    scientific_total_score: float
    oa_hint: bool
    reused_m3_2_state: bool = False
    access_status: str
    artifact_status: str
    outcome: BackfillOutcome
    artifact_id: str | None = None
    artifact_local_path: str | None = None
    error: str | None = None


class AcquisitionAwareSelectedWork(StrictModel):
    work_id: str
    title: str
    doi: str | None = None
    year: int | None = None
    venue: str | None = None
    matched_axes: list[str] = Field(default_factory=list)
    primary_quota_axis: str | None = None
    scientific_total_score: float
    source: Literal["retained_existing_m3", "m3_2_backfill"]
    artifact_id: str
    artifact_local_path: str
    artifact_sha256: str | None = None
    access_status: str
    acquisition_status: Literal["downloaded_main"] = "downloaded_main"


class AcquisitionAwareBackfillReport(StrictModel):
    schema_version: Literal[
        "acquisition-aware-backfill-report-v1"
    ] = "acquisition-aware-backfill-report-v1"
    backfill_id: str
    policy_id: str
    profile_id: str
    source_catalog_id: str
    source_m2_1_selection_report_path: str
    source_m3_report_path: str

    target_total: int
    starting_selected_count: int
    starting_downloaded_main_count: int
    starting_download_failed_count: int
    starting_not_attempted_count: int

    quality_pass_candidate_count: int
    unused_quality_pass_pool_count: int

    new_candidate_attempt_count: int
    backfill_downloaded_count: int
    backfill_failed_count: int
    backfill_not_attempted_count: int

    retained_existing_download_count: int
    final_downloaded_main_count: int
    target_reached: bool

    axis_quota_targets: dict[str, int] = Field(default_factory=dict)
    initial_axis_downloaded_counts: dict[str, int] = Field(default_factory=dict)
    final_axis_downloaded_counts: dict[str, int] = Field(default_factory=dict)
    final_unfilled_axis_quotas: dict[str, int] = Field(default_factory=dict)

    candidate_pool_exhausted: bool
    max_attempt_limit_reached: bool = False
    unattempted_quality_pass_count: int = 0

    final_selected_work_ids: list[str] = Field(default_factory=list)
    scientific_quality_gate_weakened: Literal[False] = False
    oa_added_to_scientific_score: Literal[False] = False
    paywall_bypass_attempted: Literal[False] = False
    positive_evidence_promotion_performed: Literal[False] = False
