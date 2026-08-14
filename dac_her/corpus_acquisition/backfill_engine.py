from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Callable

from dac_her.corpus_acquisition.access_contracts import (
    AccessResolution,
    SourceArtifact,
)
from dac_her.corpus_acquisition.backfill_contracts import (
    AcquisitionAwareBackfillPolicy,
    AcquisitionAwareSelectedWork,
    BackfillAttempt,
)
from dac_her.corpus_acquisition.contracts import (
    AcquisitionProfile,
    CandidateAssessment,
    SelectedCorpusWork,
)
from dac_her.corpus_acquisition.quality_contracts import (
    CorpusQualityAssessment,
)
from dac_her.literature_catalog_contracts import (
    CatalogWork,
)


AcquireFn = Callable[
    [CatalogWork],
    tuple[AccessResolution, SourceArtifact, bool],
]


def _resolve_local_path(
    value: str,
    *,
    project_root: Path,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def validate_downloaded_artifact(
    artifact: SourceArtifact,
    *,
    project_root: Path,
) -> None:
    if artifact.status != "downloaded":
        raise ValueError(
            f"Artifact is not downloaded: {artifact.artifact_id}"
        )
    if not artifact.local_path:
        raise ValueError(
            f"Downloaded artifact lacks local_path: {artifact.artifact_id}"
        )
    path = _resolve_local_path(
        artifact.local_path,
        project_root=project_root,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Downloaded artifact file missing: {path}"
        )
    if artifact.sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise RuntimeError(
                f"Downloaded artifact SHA drift: {artifact.artifact_id}: "
                f"{digest} != {artifact.sha256}"
            )


def oa_hint(
    *,
    work: CatalogWork,
    assessment: CandidateAssessment,
) -> bool:
    return bool(
        assessment.open_access_available
        or str(work.open_access_url or "").strip()
    )


def candidate_rank_key(
    *,
    work: CatalogWork,
    assessment: CandidateAssessment,
    policy: AcquisitionAwareBackfillPolicy,
) -> tuple:
    # Scientific score is always first. OA is at most a tie-break.
    hint = (
        int(oa_hint(work=work, assessment=assessment))
        if policy.oa_hint_tiebreak_only
        else 0
    )
    return (
        -float(assessment.total_score),
        -hint,
        work.title.casefold(),
        work.work_id,
    )


def choose_most_constrained_axis(
    *,
    deficits: dict[str, int],
    available_work_ids: set[str],
    assessment_map: dict[str, CandidateAssessment],
    axis_order: list[str],
) -> str | None:
    order_index = {
        axis_id: index
        for index, axis_id in enumerate(axis_order)
    }
    rows = []
    for axis_id, deficit in deficits.items():
        if deficit <= 0:
            continue
        options = sum(
            axis_id in assessment_map[work_id].matched_axes
            for work_id in available_work_ids
        )
        if options <= 0:
            continue
        # Prefer the axis with fewer candidates per remaining slot.
        rows.append(
            (
                options / float(deficit),
                options,
                order_index.get(axis_id, 10**9),
                axis_id,
            )
        )
    if not rows:
        return None
    rows.sort()
    return rows[0][-1]


def build_selected_record(
    *,
    work: CatalogWork,
    assessment: CandidateAssessment,
    primary_axis: str | None,
    source: str,
    resolution: AccessResolution,
    artifact: SourceArtifact,
) -> AcquisitionAwareSelectedWork:
    if artifact.status != "downloaded" or not artifact.local_path:
        raise ValueError("Selected record requires downloaded main artifact")
    return AcquisitionAwareSelectedWork(
        work_id=work.work_id,
        title=work.title,
        doi=work.doi,
        year=work.year,
        venue=work.venue,
        matched_axes=list(assessment.matched_axes),
        primary_quota_axis=primary_axis,
        scientific_total_score=float(assessment.total_score),
        source=source,
        artifact_id=artifact.artifact_id,
        artifact_local_path=artifact.local_path,
        artifact_sha256=artifact.sha256,
        access_status=resolution.status,
        acquisition_status="downloaded_main",
    )


def run_acquisition_aware_backfill(
    *,
    profile: AcquisitionProfile,
    policy: AcquisitionAwareBackfillPolicy,
    work_map: dict[str, CatalogWork],
    assessment_map: dict[str, CandidateAssessment],
    quality_map: dict[str, CorpusQualityAssessment],
    starting_selected: list[SelectedCorpusWork],
    starting_resolution_map: dict[str, AccessResolution],
    starting_artifact_map: dict[str, SourceArtifact],
    acquire_fn: AcquireFn,
    project_root: Path,
    progress_callback=None,
) -> tuple[
    list[AcquisitionAwareSelectedWork],
    list[BackfillAttempt],
    dict[str, AccessResolution],
    dict[str, SourceArtifact],
    dict[str, int],
    dict[str, int],
    bool,
]:
    target_total = profile.selection.target_total
    axis_order = [axis.axis_id for axis in profile.axes]
    axis_targets = {
        axis.axis_id: axis.target_selected
        for axis in profile.axes
    }

    selected_records: list[AcquisitionAwareSelectedWork] = []
    final_resolutions: dict[str, AccessResolution] = {}
    final_artifacts: dict[str, SourceArtifact] = {}
    used_work_ids: set[str] = set()
    exhausted_work_ids: set[str] = set()

    initial_counts = {
        axis_id: 0 for axis_id in axis_order
    }

    for row in starting_selected:
        quality = quality_map.get(row.work_id)
        if (
            quality is None
            or quality.status != policy.required_quality_status
        ):
            raise ValueError(
                f"M2.1 selected work is not quality=pass: {row.work_id}"
            )

        artifact = starting_artifact_map.get(row.work_id)
        resolution = starting_resolution_map.get(row.work_id)

        if (
            artifact is not None
            and artifact.status == "downloaded"
            and resolution is not None
            and policy.reuse_existing_downloaded_main
        ):
            validate_downloaded_artifact(
                artifact,
                project_root=project_root,
            )
            assessment = assessment_map[row.work_id]
            work = work_map[row.work_id]
            selected_records.append(
                build_selected_record(
                    work=work,
                    assessment=assessment,
                    primary_axis=row.primary_quota_axis,
                    source="retained_existing_m3",
                    resolution=resolution,
                    artifact=artifact,
                )
            )
            final_resolutions[row.work_id] = resolution
            final_artifacts[row.work_id] = artifact
            used_work_ids.add(row.work_id)
            if row.primary_quota_axis in initial_counts:
                initial_counts[row.primary_quota_axis] += 1
        elif policy.treat_existing_non_downloaded_as_exhausted:
            exhausted_work_ids.add(row.work_id)

    final_counts = dict(initial_counts)

    quality_pass_ids = {
        work_id
        for work_id, quality in quality_map.items()
        if (
            quality.status == policy.required_quality_status
            and work_id in assessment_map
            and assessment_map[work_id].eligibility_status == "eligible"
            and work_id in work_map
        )
    }

    available = (
        quality_pass_ids
        - used_work_ids
        - exhausted_work_ids
    )

    ranked = sorted(
        available,
        key=lambda work_id: candidate_rank_key(
            work=work_map[work_id],
            assessment=assessment_map[work_id],
            policy=policy,
        ),
    )
    rank_position = {
        work_id: index
        for index, work_id in enumerate(ranked)
    }

    attempts: list[BackfillAttempt] = []
    max_attempt_reached = False

    def attempt_work(
        work_id: str,
        *,
        phase: str,
        requested_axis: str | None,
    ) -> bool:
        nonlocal max_attempt_reached

        if (
            policy.max_new_candidate_attempts is not None
            and len(attempts) >= policy.max_new_candidate_attempts
        ):
            max_attempt_reached = True
            return False

        work = work_map[work_id]
        assessment = assessment_map[work_id]
        resolution, artifact, reused_state = acquire_fn(work)

        if artifact.status == "downloaded":
            validate_downloaded_artifact(
                artifact,
                project_root=project_root,
            )
            outcome = "downloaded"
        elif artifact.status == "download_failed":
            outcome = "download_failed"
        else:
            outcome = "not_attempted"

        attempts.append(
            BackfillAttempt(
                attempt_index=len(attempts) + 1,
                work_id=work_id,
                title=work.title,
                requested_axis=requested_axis,
                phase=phase,
                scientific_total_score=float(
                    assessment.total_score
                ),
                oa_hint=oa_hint(
                    work=work,
                    assessment=assessment,
                ),
                reused_m3_2_state=reused_state,
                access_status=resolution.status,
                artifact_status=artifact.status,
                outcome=outcome,
                artifact_id=artifact.artifact_id,
                artifact_local_path=artifact.local_path,
                error=artifact.error,
            )
        )
        exhausted_work_ids.add(work_id)

        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "m3_2_attempt",
                    "current": len(attempts),
                    "target_total": target_total,
                    "usable": len(selected_records),
                    "phase": phase,
                    "axis": requested_axis,
                    "work_id": work_id,
                    "title": work.title,
                    "artifact_status": artifact.status,
                    "access_status": resolution.status,
                    "reused_state": reused_state,
                }
            )

        if artifact.status != "downloaded":
            return False

        primary_axis = requested_axis
        selected_records.append(
            build_selected_record(
                work=work,
                assessment=assessment,
                primary_axis=primary_axis,
                source="m3_2_backfill",
                resolution=resolution,
                artifact=artifact,
            )
        )
        final_resolutions[work_id] = resolution
        final_artifacts[work_id] = artifact
        used_work_ids.add(work_id)
        if primary_axis in final_counts:
            final_counts[primary_axis] += 1
        return True

    # Phase 1: fill acquisition-adjusted axis quota deficits.
    if policy.preserve_axis_quotas:
        while len(selected_records) < target_total:
            deficits = {
                axis_id: max(
                    0,
                    axis_targets[axis_id]
                    - final_counts[axis_id],
                )
                for axis_id in axis_order
            }
            if not any(deficits.values()):
                break

            remaining_available = (
                quality_pass_ids
                - used_work_ids
                - exhausted_work_ids
            )
            axis_id = choose_most_constrained_axis(
                deficits=deficits,
                available_work_ids=remaining_available,
                assessment_map=assessment_map,
                axis_order=axis_order,
            )
            if axis_id is None:
                break

            axis_candidates = [
                work_id
                for work_id in remaining_available
                if axis_id
                in assessment_map[work_id].matched_axes
            ]
            axis_candidates.sort(
                key=lambda work_id: rank_position.get(
                    work_id,
                    10**9,
                )
            )
            if not axis_candidates:
                break

            attempted = attempt_work(
                axis_candidates[0],
                phase="quota",
                requested_axis=axis_id,
            )
            if max_attempt_reached:
                break

    # Phase 2: maximize usable corpus size without pretending global-fill
    # papers satisfy an unfilled quota.
    if (
        policy.global_fill_after_quota_attempts
        and not max_attempt_reached
    ):
        while len(selected_records) < target_total:
            remaining_available = [
                work_id
                for work_id in quality_pass_ids
                if (
                    work_id not in used_work_ids
                    and work_id not in exhausted_work_ids
                )
            ]
            if not remaining_available:
                break
            remaining_available.sort(
                key=lambda work_id: rank_position.get(
                    work_id,
                    10**9,
                )
            )
            attempt_work(
                remaining_available[0],
                phase="global_fill",
                requested_axis=None,
            )
            if max_attempt_reached:
                break

    return (
        selected_records,
        attempts,
        final_resolutions,
        final_artifacts,
        initial_counts,
        final_counts,
        max_attempt_reached,
    )
