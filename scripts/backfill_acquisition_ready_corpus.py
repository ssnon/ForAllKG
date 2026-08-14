from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dac_her.corpus_acquisition.access_contracts import (
    AccessResolution,
    CorpusSourceAcquisitionReport,
    SourceArtifact,
)
from dac_her.corpus_acquisition.artifact_acquisition import (
    MainArtifactDownloader,
)
from dac_her.corpus_acquisition.backfill_compat import (
    compatible_acquisition_report,
    compatible_selected_works,
    compatible_selection_report,
)
from dac_her.corpus_acquisition.backfill_contracts import (
    AcquisitionAwareBackfillReport,
)
from dac_her.corpus_acquisition.backfill_engine import (
    run_acquisition_aware_backfill,
)
from dac_her.corpus_acquisition.backfill_policy import (
    load_acquisition_backfill_policy,
)
from dac_her.corpus_acquisition.backfill_state import (
    atomic_write_json,
    write_jsonl,
)
from dac_her.corpus_acquisition.contracts import (
    CandidateAssessment,
    CorpusSelectionReport,
    SelectedCorpusWork,
)
from dac_her.corpus_acquisition.oa_resolution import (
    OpenAccessResolver,
)
from dac_her.corpus_acquisition.profile import (
    load_acquisition_profile,
)
from dac_her.corpus_acquisition.progress import (
    compact_text,
    progress_prefix,
)
from dac_her.corpus_acquisition.quality_contracts import (
    CorpusQualityAssessment,
    CorpusQualityGateReport,
)
from dac_her.corpus_acquisition.source_policy import (
    load_source_acquisition_policy,
)
from dac_her.corpus_acquisition.source_state import (
    load_work_state,
    safe_state_name,
    write_work_state,
)
from dac_her.literature_catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path, model) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M3.2 acquisition-aware backfill. Reuse verified main PDFs from "
            "the quality-gated M3 run, then fill missing quota/total slots "
            "only from the remaining quality=pass scientific candidate pool. "
            "OA availability never changes scientific eligibility or score."
        )
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--backfill-policy", required=True, type=Path)
    parser.add_argument("--source-policy", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--m2-assessments", required=True, type=Path)
    parser.add_argument("--quality-assessments", required=True, type=Path)
    parser.add_argument("--quality-gate-report", required=True, type=Path)
    parser.add_argument("--m2-1-selected-works", required=True, type=Path)
    parser.add_argument("--m2-1-selection-report", required=True, type=Path)
    parser.add_argument("--m3-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backfill-id", required=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry M3.2 candidate states whose previous artifact status "
            "was download_failed. Existing M3 failures remain exhausted."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    profile = load_acquisition_profile(args.profile)
    backfill_policy = load_acquisition_backfill_policy(
        args.backfill_policy
    )
    source_policy = load_source_acquisition_policy(
        args.source_policy
    )

    packet = LiteratureCatalogPacket.model_validate_json(
        args.catalog.read_text(encoding="utf-8")
    )
    assessments = _read_jsonl(
        args.m2_assessments,
        CandidateAssessment,
    )
    quality_rows = _read_jsonl(
        args.quality_assessments,
        CorpusQualityAssessment,
    )
    quality_report = CorpusQualityGateReport.model_validate_json(
        args.quality_gate_report.read_text(encoding="utf-8")
    )
    starting_selected = _read_jsonl(
        args.m2_1_selected_works,
        SelectedCorpusWork,
    )
    starting_selection_report = (
        CorpusSelectionReport.model_validate_json(
            args.m2_1_selection_report.read_text(
                encoding="utf-8"
            )
        )
    )

    m3_report_path = args.m3_dir / "acquisition_report.json"
    if not m3_report_path.exists():
        raise FileNotFoundError(
            f"Completed robust M3 report required: {m3_report_path}"
        )
    starting_m3_report = (
        CorpusSourceAcquisitionReport.model_validate_json(
            m3_report_path.read_text(encoding="utf-8")
        )
    )
    starting_resolutions = _read_jsonl(
        args.m3_dir / "access_resolutions.jsonl",
        AccessResolution,
    )
    starting_artifacts = _read_jsonl(
        args.m3_dir / "artifacts.jsonl",
        SourceArtifact,
    )

    if packet.acquisition_profile_id != profile.profile_id:
        raise ValueError("Catalog/profile mismatch")
    if quality_report.profile_id != profile.profile_id:
        raise ValueError("Quality report/profile mismatch")
    if quality_report.source_catalog_id != packet.catalog_id:
        raise ValueError("Quality report/catalog mismatch")
    if starting_selection_report.profile_id != profile.profile_id:
        raise ValueError("M2.1 selection/profile mismatch")
    if (
        starting_selection_report.source_catalog_id
        != packet.catalog_id
    ):
        raise ValueError("M2.1 selection/catalog mismatch")
    if (
        starting_selection_report.selected_work_ids
        != [row.work_id for row in starting_selected]
    ):
        raise ValueError(
            "M2.1 selected_works does not match M2.1 selection report"
        )
    if starting_m3_report.source_profile_id != profile.profile_id:
        raise ValueError("M3/profile mismatch")
    if starting_m3_report.source_catalog_id != packet.catalog_id:
        raise ValueError("M3/catalog mismatch")

    work_map: dict[str, CatalogWork] = {
        row.work_id: row
        for row in packet.works
    }
    assessment_map = {
        row.work_id: row
        for row in assessments
    }
    quality_map = {
        row.work_id: row
        for row in quality_rows
    }
    if len(assessment_map) != len(assessments):
        raise ValueError("Duplicate M2 assessment work_id")
    if len(quality_map) != len(quality_rows):
        raise ValueError("Duplicate quality assessment work_id")

    starting_resolution_map = {
        row.work_id: row
        for row in starting_resolutions
    }
    starting_artifact_map = {
        row.work_id: row
        for row in starting_artifacts
    }

    output = args.output_dir
    state_root = output / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    resolver = OpenAccessResolver(source_policy)
    downloader = MainArtifactDownloader(source_policy)

    def acquire(work: CatalogWork):
        path = state_root / safe_state_name(work.work_id)
        prior = load_work_state(path)
        if prior is not None:
            prior_resolution, prior_artifact = prior
            if (
                prior_artifact.status != "download_failed"
                or not args.retry_failed
            ):
                return (
                    prior_resolution,
                    prior_artifact,
                    True,
                )

        resolution = resolver.resolve(work)
        artifact = downloader.acquire(
            work=work,
            resolution=resolution,
            output_root=output,
        )
        write_work_state(
            path=path,
            resolution=resolution,
            artifact=artifact,
        )
        return resolution, artifact, False

    def progress(event: dict[str, Any]) -> None:
        print(
            progress_prefix(
                "M3.2 attempt",
                int(event["current"]),
                max(
                    int(event["current"]),
                    int(event["target_total"]),
                ),
            ),
            f"usable={event['usable']}/{event['target_total']}",
            f"phase={event['phase']}",
            f"axis={event.get('axis') or '-'}",
            f"access={event['access_status']}",
            f"artifact={event['artifact_status']}",
            (
                "resume"
                if event.get("reused_state")
                else "network"
            ),
            compact_text(
                str(event["title"]),
                max_length=48,
            ),
            flush=True,
        )

    (
        selected,
        attempts,
        final_resolution_map,
        final_artifact_map,
        initial_axis_counts,
        final_axis_counts,
        max_attempt_reached,
    ) = run_acquisition_aware_backfill(
        profile=profile,
        policy=backfill_policy,
        work_map=work_map,
        assessment_map=assessment_map,
        quality_map=quality_map,
        starting_selected=starting_selected,
        starting_resolution_map=starting_resolution_map,
        starting_artifact_map=starting_artifact_map,
        acquire_fn=acquire,
        project_root=PROJECT_ROOT,
        progress_callback=progress,
    )

    # Deterministic final order: original M2.1 order first when retained,
    # then backfill works by axis order/scientific score/work_id.
    starting_order = {
        row.work_id: index
        for index, row in enumerate(starting_selected)
    }
    axis_order = {
        axis.axis_id: index
        for index, axis in enumerate(profile.axes)
    }
    selected.sort(
        key=lambda row: (
            0
            if row.source == "retained_existing_m3"
            else 1,
            starting_order.get(row.work_id, 10**9),
            axis_order.get(
                row.primary_quota_axis,
                10**9,
            ),
            -row.scientific_total_score,
            row.work_id,
        )
    )

    final_ids = [row.work_id for row in selected]
    final_resolutions = [
        final_resolution_map[work_id]
        for work_id in final_ids
    ]
    final_artifacts = [
        final_artifact_map[work_id]
        for work_id in final_ids
    ]

    compatible_selected = compatible_selected_works(
        selected=selected,
        work_map=work_map,
    )
    compatible_selection = compatible_selection_report(
        profile=profile,
        packet=packet,
        assessments=assessments,
        selected=selected,
    )

    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_works.jsonl"
    selection_report_path = output / "selection_report.json"

    write_jsonl(
        output / "backfill_selected_works.jsonl",
        selected,
    )
    write_jsonl(
        output / "backfill_attempts.jsonl",
        attempts,
    )
    write_jsonl(
        selected_path,
        compatible_selected,
    )
    atomic_write_json(
        selection_report_path,
        compatible_selection,
    )
    write_jsonl(
        output / "access_resolutions.jsonl",
        final_resolutions,
    )
    write_jsonl(
        output / "artifacts.jsonl",
        final_artifacts,
    )

    compatible_acquisition = compatible_acquisition_report(
        acquisition_id=args.backfill_id,
        profile_id=profile.profile_id,
        packet=packet,
        source_selection_report_path=selection_report_path,
        policy_id=source_policy.policy_id,
        output_root=output,
        resolutions=final_resolutions,
        artifacts=final_artifacts,
        upstream_report=starting_m3_report,
    )
    atomic_write_json(
        output / "acquisition_report.json",
        compatible_acquisition,
    )

    quality_pass_ids = {
        row.work_id
        for row in quality_rows
        if (
            row.status == "pass"
            and row.work_id in assessment_map
            and assessment_map[row.work_id].eligibility_status
            == "eligible"
        )
    }
    starting_ids = {
        row.work_id for row in starting_selected
    }
    unused_pool_initial = quality_pass_ids - starting_ids

    attempted_ids = {
        row.work_id for row in attempts
    }
    retained_ids = {
        row.work_id
        for row in selected
        if row.source == "retained_existing_m3"
    }
    final_id_set = set(final_ids)
    remaining_unattempted = (
        quality_pass_ids
        - starting_ids
        - attempted_ids
        - final_id_set
    )

    axis_targets = {
        axis.axis_id: axis.target_selected
        for axis in profile.axes
    }
    final_unfilled = {
        axis_id: max(
            0,
            axis_targets[axis_id]
            - final_axis_counts.get(axis_id, 0),
        )
        for axis_id in axis_targets
        if (
            axis_targets[axis_id]
            - final_axis_counts.get(axis_id, 0)
            > 0
        )
    }

    starting_artifact_counts = Counter(
        row.status for row in starting_artifacts
    )
    attempt_counts = Counter(
        row.outcome for row in attempts
    )

    report = AcquisitionAwareBackfillReport(
        backfill_id=args.backfill_id,
        policy_id=backfill_policy.policy_id,
        profile_id=profile.profile_id,
        source_catalog_id=packet.catalog_id,
        source_m2_1_selection_report_path=str(
            args.m2_1_selection_report
        ),
        source_m3_report_path=str(m3_report_path),
        target_total=profile.selection.target_total,
        starting_selected_count=len(starting_selected),
        starting_downloaded_main_count=starting_artifact_counts[
            "downloaded"
        ],
        starting_download_failed_count=starting_artifact_counts[
            "download_failed"
        ],
        starting_not_attempted_count=starting_artifact_counts[
            "not_attempted"
        ],
        quality_pass_candidate_count=len(quality_pass_ids),
        unused_quality_pass_pool_count=len(unused_pool_initial),
        new_candidate_attempt_count=len(attempts),
        backfill_downloaded_count=attempt_counts["downloaded"],
        backfill_failed_count=attempt_counts["download_failed"],
        backfill_not_attempted_count=attempt_counts[
            "not_attempted"
        ],
        retained_existing_download_count=len(retained_ids),
        final_downloaded_main_count=len(selected),
        target_reached=(
            len(selected)
            >= profile.selection.target_total
        ),
        axis_quota_targets=axis_targets,
        initial_axis_downloaded_counts=initial_axis_counts,
        final_axis_downloaded_counts=final_axis_counts,
        final_unfilled_axis_quotas=final_unfilled,
        candidate_pool_exhausted=(
            len(selected) < profile.selection.target_total
            and not remaining_unattempted
        ),
        max_attempt_limit_reached=max_attempt_reached,
        unattempted_quality_pass_count=len(
            remaining_unattempted
        ),
        final_selected_work_ids=final_ids,
        scientific_quality_gate_weakened=False,
        oa_added_to_scientific_score=False,
        paywall_bypass_attempted=False,
        positive_evidence_promotion_performed=False,
    )
    atomic_write_json(
        output / "backfill_report.json",
        report,
    )

    print()
    print("Generic corpus acquisition M3.2 complete")
    print(
        "Starting usable main PDFs:",
        report.starting_downloaded_main_count,
        "/",
        report.starting_selected_count,
    )
    print(
        "Quality-pass candidate pool:",
        report.quality_pass_candidate_count,
        f"(unused after M2.1={report.unused_quality_pass_pool_count})",
    )
    print(
        "Backfill:",
        f"attempts={report.new_candidate_attempt_count}",
        f"downloaded={report.backfill_downloaded_count}",
        f"failed={report.backfill_failed_count}",
        f"not_attempted={report.backfill_not_attempted_count}",
    )
    print(
        "Final usable corpus:",
        report.final_downloaded_main_count,
        "/ target",
        report.target_total,
        f"reached={report.target_reached}",
    )
    print(
        "Unfilled usable axis quotas:",
        report.final_unfilled_axis_quotas,
    )
    print(
        "Pool exhausted:",
        report.candidate_pool_exhausted,
    )
    print(
        "Unattempted quality-pass candidates:",
        report.unattempted_quality_pass_count,
    )
    print(
        "Scientific quality gate weakened:",
        report.scientific_quality_gate_weakened,
    )
    print(
        "OA added to scientific score:",
        report.oa_added_to_scientific_score,
    )
    print(
        "Paywall bypass attempted:",
        report.paywall_bypass_attempted,
    )
    print("Downstream selected works:", selected_path)
    print(
        "Downstream selection report:",
        selection_report_path,
    )
    print(
        "Downstream M3-compatible dir:",
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
