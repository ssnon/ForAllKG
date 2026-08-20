from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from pipeline_core.literature.acquisition.access_contracts import AccessResolution, CorpusSourceAcquisitionReport, SourceArtifact
from pipeline_core.literature.acquisition.backfill_contracts import AcquisitionAwareSelectedWork
from pipeline_core.literature.acquisition.contracts import AcquisitionProfile, CandidateAssessment, CorpusSelectionReport, SelectedCorpusWork
from pipeline_core.literature.catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


def compatible_selected_works(
    *,
    selected: list[AcquisitionAwareSelectedWork],
    work_map: dict[str, CatalogWork],
) -> list[SelectedCorpusWork]:
    rows = []
    for row in selected:
        work = work_map[row.work_id]
        rows.append(
            SelectedCorpusWork(
                work_id=row.work_id,
                title=row.title,
                doi=row.doi,
                year=row.year,
                venue=row.venue,
                open_access_url=work.open_access_url,
                matched_axes=list(row.matched_axes),
                primary_quota_axis=row.primary_quota_axis,
                total_score=row.scientific_total_score,
                acquisition_status="selected_metadata_only",
            )
        )
    return rows


def compatible_selection_report(
    *,
    profile: AcquisitionProfile,
    packet: LiteratureCatalogPacket,
    assessments: list[CandidateAssessment],
    selected: list[AcquisitionAwareSelectedWork],
) -> CorpusSelectionReport:
    axis_targets = {
        axis.axis_id: axis.target_selected
        for axis in profile.axes
    }
    primary_counts = Counter(
        row.primary_quota_axis
        for row in selected
        if row.primary_quota_axis is not None
    )
    axis_candidate_counts = {
        axis.axis_id: sum(
            (
                assessment.eligibility_status == "eligible"
                and axis.axis_id in assessment.matched_axes
            )
            for assessment in assessments
        )
        for axis in profile.axes
    }
    unfilled = {
        axis_id: max(
            0,
            target - primary_counts[axis_id],
        )
        for axis_id, target in axis_targets.items()
        if target - primary_counts[axis_id] > 0
    }
    return CorpusSelectionReport(
        profile_id=profile.profile_id,
        source_catalog_id=packet.catalog_id,
        candidate_count=len(assessments),
        eligible_count=sum(
            row.eligibility_status == "eligible"
            for row in assessments
        ),
        manual_review_count=sum(
            row.eligibility_status == "manual_review"
            for row in assessments
        ),
        excluded_count=sum(
            row.eligibility_status == "excluded"
            for row in assessments
        ),
        selected_count=len(selected),
        target_total=profile.selection.target_total,
        axis_candidate_counts=axis_candidate_counts,
        axis_quota_targets=axis_targets,
        axis_primary_selected_counts={
            axis_id: primary_counts[axis_id]
            for axis_id in axis_targets
        },
        unfilled_axis_quotas=unfilled,
        selected_work_ids=[
            row.work_id for row in selected
        ],
        policy_notes=[
            "Compatibility report for M3.2 acquisition-aware backfill.",
            (
                "Only quality-gate-pass works with verified downloaded main "
                "artifacts are included."
            ),
            (
                "acquisition_status remains selected_metadata_only solely "
                "for compatibility with the existing downstream contract; "
                "M3.2 backfill_selected_works.jsonl carries downloaded_main."
            ),
        ],
        positive_evidence_promotion_performed=False,
    )


def compatible_acquisition_report(
    *,
    acquisition_id: str,
    profile_id: str,
    packet: LiteratureCatalogPacket,
    source_selection_report_path: Path,
    policy_id: str,
    output_root: Path,
    resolutions: list[AccessResolution],
    artifacts: list[SourceArtifact],
    upstream_report: CorpusSourceAcquisitionReport,
) -> CorpusSourceAcquisitionReport:
    access_counts = Counter(row.status for row in resolutions)
    artifact_counts = Counter(row.status for row in artifacts)
    unpaywall_attempts = [
        attempt
        for row in resolutions
        for attempt in row.resolver_attempts
        if (
            attempt.resolver == "unpaywall"
            and attempt.status != "skipped"
        )
    ]
    catalog_fallback = sum(
        any(
            location.resolver == "catalog_open_access"
            for location in row.locations
        )
        for row in resolutions
    )

    all_attempts = [
        attempt
        for artifact in artifacts
        for attempt in artifact.download_attempts
    ]
    failure_reasons = Counter(
        attempt.error_code or "unknown"
        for attempt in all_attempts
        if attempt.status == "failed"
    )
    host_stats: dict[str, dict[str, int]] = {}
    for attempt in all_attempts:
        host = attempt.host or "unknown"
        stats = host_stats.setdefault(
            host,
            {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
            },
        )
        stats["attempts"] += 1
        if attempt.status == "success":
            stats["successes"] += 1
        else:
            stats["failures"] += 1

    recoveries = sum(
        artifact.status == "downloaded"
        and any(
            attempt.status == "failed"
            for attempt in artifact.download_attempts[:-1]
        )
        for artifact in artifacts
    )

    return CorpusSourceAcquisitionReport(
        acquisition_id=acquisition_id,
        source_profile_id=profile_id,
        source_catalog_id=packet.catalog_id,
        source_selection_report_path=str(
            source_selection_report_path
        ),
        policy_id=policy_id,
        selected_work_count=len(artifacts),
        access_resolved_direct_pdf_count=access_counts[
            "resolved_direct_pdf"
        ],
        access_resolved_landing_only_count=access_counts[
            "resolved_landing_only"
        ],
        access_unresolved_count=access_counts["unresolved"],
        artifact_downloaded_count=artifact_counts["downloaded"],
        artifact_download_failed_count=artifact_counts[
            "download_failed"
        ],
        artifact_not_attempted_count=artifact_counts[
            "not_attempted"
        ],
        unpaywall_attempt_count=len(unpaywall_attempts),
        unpaywall_success_count=sum(
            attempt.status == "success"
            for attempt in unpaywall_attempts
        ),
        catalog_oa_fallback_count=catalog_fallback,
        total_download_location_attempts=len(all_attempts),
        multi_location_recovery_count=recoveries,
        download_failure_reason_counts={
            key: failure_reasons[key]
            for key in sorted(failure_reasons)
        },
        download_host_stats={
            key: host_stats[key]
            for key in sorted(host_stats)
        },
        upstream_provider_query_success_count=(
            upstream_report
            .upstream_provider_query_success_count
        ),
        upstream_provider_query_execution_count=(
            upstream_report
            .upstream_provider_query_execution_count
        ),
        upstream_coverage_warning=(
            upstream_report.upstream_coverage_warning
        ),
        output_root=str(output_root),
        supplementary_discovery="deferred_to_m3_1",
        paywall_bypass_attempted=False,
        positive_evidence_promotion_performed=False,
    )
