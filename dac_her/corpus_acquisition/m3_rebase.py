from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from dac_her.corpus_acquisition.access_contracts import (
    AccessResolution,
    CorpusSourceAcquisitionReport,
    SourceArtifact,
)
from dac_her.corpus_acquisition.contracts import (
    AcquisitionProfile,
    CandidateAssessment,
    CorpusSelectionReport,
    SelectedCorpusWork,
)
from dac_her.corpus_acquisition.quality_contracts import CorpusQualityAssessment
from dac_her.corpus_acquisition.source_state import (
    atomic_write_json,
    safe_state_name,
    write_jsonl,
    write_work_state,
)
from pipeline_core.literature.catalog_contracts import LiteratureCatalogPacket


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pdf_artifact(artifact: SourceArtifact) -> Path:
    if artifact.status != "downloaded":
        raise ValueError(f"Artifact is not downloaded: {artifact.work_id}")
    if not artifact.local_path:
        raise ValueError(f"Downloaded artifact has no local_path: {artifact.work_id}")
    path = Path(artifact.local_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        if not handle.read(5).startswith(b"%PDF-"):
            raise ValueError(f"Downloaded artifact is not a PDF: {path}")
    digest = _sha256_file(path)
    if artifact.sha256 and artifact.sha256 != digest:
        raise ValueError(
            f"Artifact SHA mismatch for {artifact.work_id}: "
            f"{artifact.sha256} != {digest}"
        )
    return path


def _read_jsonl(path: Path, model) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows


def _selection_report(
    *,
    profile: AcquisitionProfile,
    packet: LiteratureCatalogPacket,
    assessments: list[CandidateAssessment],
    selected: list[SelectedCorpusWork],
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
            row.eligibility_status == "eligible"
            and axis.axis_id in row.matched_axes
            for row in assessments
        )
        for axis in profile.axes
    }
    unfilled = {
        axis_id: max(0, target - primary_counts[axis_id])
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
        selected_work_ids=[row.work_id for row in selected],
        policy_notes=[
            "Catalog-expansion M3 rebase snapshot.",
            (
                "Only previously verified downloaded main PDFs that remain "
                "eligible and quality=pass in the expanded catalog are retained."
            ),
            (
                "No network acquisition is performed by this rebase stage; "
                "new expanded-catalog candidates are left for M3.2 backfill."
            ),
        ],
        positive_evidence_promotion_performed=False,
    )


def rebase_downloaded_m3_snapshot(
    *,
    profile: AcquisitionProfile,
    packet: LiteratureCatalogPacket,
    assessments: list[CandidateAssessment],
    quality_assessments: list[CorpusQualityAssessment],
    source_m3_dir: Path,
    output_dir: Path,
    rebase_id: str,
) -> dict[str, object]:
    """Rebase verified downloaded M3 works onto an append-expanded catalog.

    This stage performs no resolver calls and no downloads.  It verifies and
    copies already-downloaded PDFs into a new self-contained M3-compatible
    snapshot whose catalog lineage points at the expanded catalog.
    """

    source_selected = _read_jsonl(
        source_m3_dir / "selected_works.jsonl",
        SelectedCorpusWork,
    )
    source_resolutions = _read_jsonl(
        source_m3_dir / "access_resolutions.jsonl",
        AccessResolution,
    )
    source_artifacts = _read_jsonl(
        source_m3_dir / "artifacts.jsonl",
        SourceArtifact,
    )
    source_report = CorpusSourceAcquisitionReport.model_validate_json(
        (source_m3_dir / "acquisition_report.json").read_text(encoding="utf-8")
    )

    if source_report.source_profile_id != profile.profile_id:
        raise ValueError("Source M3/profile mismatch")
    if packet.acquisition_profile_id != profile.profile_id:
        raise ValueError("Expanded catalog/profile mismatch")

    work_map = {row.work_id: row for row in packet.works}
    assessment_map = {row.work_id: row for row in assessments}
    quality_map = {row.work_id: row for row in quality_assessments}
    resolution_map = {row.work_id: row for row in source_resolutions}
    artifact_map = {row.work_id: row for row in source_artifacts}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "state").mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    retained_selected: list[SelectedCorpusWork] = []
    retained_resolutions: list[AccessResolution] = []
    retained_artifacts: list[SourceArtifact] = []
    dropped: list[dict[str, str]] = []

    for selected in source_selected:
        work_id = selected.work_id
        if work_id not in work_map:
            raise ValueError(
                "Append-only catalog invariant violated; source M3 work is "
                f"missing from expanded catalog: {work_id}"
            )
        assessment = assessment_map.get(work_id)
        quality = quality_map.get(work_id)
        resolution = resolution_map.get(work_id)
        artifact = artifact_map.get(work_id)

        if assessment is None or quality is None:
            raise ValueError(
                f"Expanded M2/M2.1 rows missing retained work: {work_id}"
            )
        if assessment.eligibility_status != "eligible":
            dropped.append({"work_id": work_id, "reason": "m2_not_eligible"})
            continue
        if quality.status != "pass":
            dropped.append({"work_id": work_id, "reason": "quality_not_pass"})
            continue
        if resolution is None or artifact is None:
            dropped.append({"work_id": work_id, "reason": "source_m3_state_missing"})
            continue
        if artifact.status != "downloaded":
            dropped.append({"work_id": work_id, "reason": "source_not_downloaded"})
            continue

        source_pdf = _verify_pdf_artifact(artifact)
        state_stem = safe_state_name(work_id).removesuffix(".json")
        destination = output_dir / "artifacts" / state_stem / "main.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_pdf, destination)
        copied_sha = _sha256_file(destination)
        if artifact.sha256 and copied_sha != artifact.sha256:
            raise RuntimeError(
                f"Copied artifact SHA mismatch for {work_id}: {copied_sha}"
            )

        work = work_map[work_id]
        primary_axis = selected.primary_quota_axis
        if primary_axis and primary_axis not in assessment.matched_axes:
            raise ValueError(
                "Frozen primary quota axis is no longer supported by expanded "
                f"M2 assessment for {work_id}: {primary_axis}"
            )

        rebased_selected = SelectedCorpusWork(
            work_id=work_id,
            title=work.title,
            doi=work.doi,
            year=work.year,
            venue=work.venue,
            open_access_url=work.open_access_url,
            matched_axes=list(assessment.matched_axes),
            primary_quota_axis=primary_axis,
            total_score=assessment.total_score,
            acquisition_status="selected_metadata_only",
        )
        rebased_artifact = artifact.model_copy(
            update={
                "local_path": str(destination.resolve()),
                "sha256": copied_sha,
                "byte_count": destination.stat().st_size,
                "acquisition_method": "catalog_expansion_rebase_verified_pdf",
            }
        )

        retained_selected.append(rebased_selected)
        retained_resolutions.append(resolution)
        retained_artifacts.append(rebased_artifact)
        write_work_state(
            path=output_dir / "state" / safe_state_name(work_id),
            resolution=resolution,
            artifact=rebased_artifact,
        )

    if not retained_selected:
        raise ValueError("Catalog expansion rebase retained zero downloaded papers")

    selection_report = _selection_report(
        profile=profile,
        packet=packet,
        assessments=assessments,
        selected=retained_selected,
    )

    access_counts = Counter(row.status for row in retained_resolutions)
    artifact_counts = Counter(row.status for row in retained_artifacts)
    acquisition_report = CorpusSourceAcquisitionReport(
        acquisition_id=rebase_id,
        source_profile_id=profile.profile_id,
        source_catalog_id=packet.catalog_id,
        source_selection_report_path=str(
            (output_dir / "selection_report.json").resolve()
        ),
        policy_id=source_report.policy_id,
        selected_work_count=len(retained_selected),
        access_resolved_direct_pdf_count=access_counts["resolved_direct_pdf"],
        access_resolved_landing_only_count=access_counts["resolved_landing_only"],
        access_unresolved_count=access_counts["unresolved"],
        artifact_downloaded_count=artifact_counts["downloaded"],
        artifact_download_failed_count=artifact_counts["download_failed"],
        artifact_not_attempted_count=artifact_counts["not_attempted"],
        upstream_provider_query_success_count=(
            source_report.upstream_provider_query_success_count
        ),
        upstream_provider_query_execution_count=(
            source_report.upstream_provider_query_execution_count
        ),
        upstream_coverage_warning=source_report.upstream_coverage_warning,
        output_root=str(output_dir.resolve()),
        supplementary_discovery="deferred_to_m3_1",
        paywall_bypass_attempted=False,
        positive_evidence_promotion_performed=False,
    )

    write_jsonl(output_dir / "selected_works.jsonl", retained_selected)
    write_jsonl(output_dir / "access_resolutions.jsonl", retained_resolutions)
    write_jsonl(output_dir / "artifacts.jsonl", retained_artifacts)
    atomic_write_json(output_dir / "selection_report.json", selection_report)
    atomic_write_json(output_dir / "acquisition_report.json", acquisition_report)

    report: dict[str, object] = {
        "schema_version": "catalog-expansion-m3-rebase-report-v1",
        "rebase_id": rebase_id,
        "source_m3_dir": str(source_m3_dir.resolve()),
        "source_catalog_id": source_report.source_catalog_id,
        "expanded_catalog_id": packet.catalog_id,
        "source_selected_count": len(source_selected),
        "retained_downloaded_count": len(retained_selected),
        "dropped_count": len(dropped),
        "dropped": dropped,
        "network_acquisition_performed": False,
        "pdf_bytes_copied": sum(
            int(row.byte_count or 0) for row in retained_artifacts
        ),
        "retained_work_ids": [row.work_id for row in retained_selected],
        "paywall_bypass_attempted": False,
        "positive_evidence_promotion_performed": False,
    }
    atomic_write_json(output_dir / "rebase_report.json", report)
    return report
