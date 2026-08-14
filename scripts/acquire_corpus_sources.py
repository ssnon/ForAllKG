from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dac_her.corpus_acquisition.access_contracts import (
    CorpusSourceAcquisitionReport,
    SourceArtifact,
)
from dac_her.corpus_acquisition.artifact_acquisition import (
    MainArtifactDownloader,
)
from dac_her.corpus_acquisition.oa_resolution import (
    OpenAccessResolver,
)
from dac_her.corpus_acquisition.progress import (
    compact_text,
    progress_prefix,
)
from dac_her.corpus_acquisition.source_policy import (
    load_source_acquisition_policy,
)
from dac_her.corpus_acquisition.source_state import (
    atomic_write_json,
    load_work_state,
    safe_state_name,
    write_jsonl,
    write_work_state,
)
from dac_her.corpus_acquisition.contracts import (
    CorpusSelectionReport,
    SelectedCorpusWork,
)
from dac_her.literature_catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


def _read_jsonl(path: Path, model) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(model.model_validate_json(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic Corpus Acquisition M3: resolve publicly available "
            "open-access main-PDF locations and acquire verified source "
            "artifacts. No paywall bypass and no positive KG evidence "
            "promotion are performed."
        )
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--selected-works", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--source-policy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--acquisition-id", required=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry work states whose previous main artifact status was "
            "download_failed. Downloaded/not-attempted states remain resumable."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = LiteratureCatalogPacket.model_validate_json(
        args.catalog.read_text(encoding="utf-8")
    )
    selected = _read_jsonl(
        args.selected_works,
        SelectedCorpusWork,
    )
    selection_report = CorpusSelectionReport.model_validate_json(
        args.selection_report.read_text(encoding="utf-8")
    )
    policy = load_source_acquisition_policy(args.source_policy)

    if packet.acquisition_profile_id != args.profile_id:
        raise ValueError(
            "Catalog/profile mismatch: "
            f"{packet.acquisition_profile_id!r} != {args.profile_id!r}"
        )
    if selection_report.profile_id != args.profile_id:
        raise ValueError("Selection report/profile mismatch")
    if selection_report.source_catalog_id != packet.catalog_id:
        raise ValueError("Selection report/catalog mismatch")

    selected_ids = [row.work_id for row in selected]
    if selected_ids != selection_report.selected_work_ids:
        raise ValueError(
            "selected_works.jsonl order/content does not match selection report"
        )
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate selected work IDs")

    work_map: dict[str, CatalogWork] = {
        work.work_id: work for work in packet.works
    }
    missing = [
        work_id
        for work_id in selected_ids
        if work_id not in work_map
    ]
    if missing:
        raise ValueError(
            f"Selected works missing from catalog: {missing[:10]!r}"
        )

    output = args.output_dir
    state_root = output / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    resolver = OpenAccessResolver(policy)
    downloader = MainArtifactDownloader(policy)

    resolutions = []
    artifacts = []
    total = len(selected)

    for index, selected_row in enumerate(selected, start=1):
        work = work_map[selected_row.work_id]
        state_path = state_root / safe_state_name(work.work_id)
        prior = load_work_state(state_path)

        should_resume = (
            prior is not None
            and (
                prior[1].status != "download_failed"
                or not args.retry_failed
            )
        )
        if should_resume:
            resolution, artifact = prior
            print(
                progress_prefix("M3", index, total),
                "resume",
                f"access={resolution.status}",
                f"artifact={artifact.status}",
                compact_text(work.title, max_length=68),
                flush=True,
            )
        else:
            print(
                progress_prefix("M3", index, total),
                "resolve",
                f"doi={work.doi or '-'}",
                compact_text(work.title, max_length=72),
                flush=True,
            )
            resolution = resolver.resolve(work)
            print(
                progress_prefix("M3", index, total),
                f"access={resolution.status}",
                f"locations={len(resolution.locations)}",
                flush=True,
            )
            artifact = downloader.acquire(
                work=work,
                resolution=resolution,
                output_root=output,
            )
            print(
                progress_prefix("M3", index, total),
                f"artifact={artifact.status}",
                (
                    f"bytes={artifact.byte_count}"
                    if artifact.byte_count is not None
                    else ""
                ),
                flush=True,
            )
            write_work_state(
                path=state_path,
                resolution=resolution,
                artifact=artifact,
            )

        resolutions.append(resolution)
        artifacts.append(artifact)

    write_jsonl(
        output / "access_resolutions.jsonl",
        resolutions,
    )
    write_jsonl(
        output / "artifacts.jsonl",
        artifacts,
    )

    access_counts = Counter(row.status for row in resolutions)
    artifact_counts = Counter(row.status for row in artifacts)
    unpaywall_attempts = [
        attempt
        for row in resolutions
        for attempt in row.resolver_attempts
        if attempt.resolver == "unpaywall"
        and attempt.status != "skipped"
    ]
    unpaywall_successes = [
        attempt
        for attempt in unpaywall_attempts
        if attempt.status == "success"
    ]
    catalog_fallback = sum(
        any(
            location.resolver == "catalog_open_access"
            for location in row.locations
        )
        for row in resolutions
    )

    all_download_attempts = [
        attempt
        for artifact in artifacts
        for attempt in artifact.download_attempts
    ]
    failure_reason_counts = Counter(
        attempt.error_code or "unknown"
        for attempt in all_download_attempts
        if attempt.status == "failed"
    )
    host_stats: dict[str, dict[str, int]] = {}
    for attempt in all_download_attempts:
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
    multi_location_recovery_count = sum(
        artifact.status == "downloaded"
        and any(
            attempt.status == "failed"
            for attempt in artifact.download_attempts[:-1]
        )
        for artifact in artifacts
    )

    successful_upstream = sum(
        execution.success for execution in packet.executions
    )
    upstream_total = len(packet.executions)
    upstream_warning = (
        upstream_total > 0
        and successful_upstream < upstream_total
    )

    report = CorpusSourceAcquisitionReport(
        acquisition_id=args.acquisition_id,
        source_profile_id=args.profile_id,
        source_catalog_id=packet.catalog_id,
        source_selection_report_path=str(args.selection_report),
        policy_id=policy.policy_id,
        selected_work_count=total,
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
        unpaywall_success_count=len(unpaywall_successes),
        catalog_oa_fallback_count=catalog_fallback,
        total_download_location_attempts=len(all_download_attempts),
        multi_location_recovery_count=multi_location_recovery_count,
        download_failure_reason_counts={
            key: failure_reason_counts[key]
            for key in sorted(failure_reason_counts)
        },
        download_host_stats={
            key: host_stats[key]
            for key in sorted(host_stats)
        },
        upstream_provider_query_success_count=successful_upstream,
        upstream_provider_query_execution_count=upstream_total,
        upstream_coverage_warning=upstream_warning,
        output_root=str(output),
        supplementary_discovery="deferred_to_m3_1",
        paywall_bypass_attempted=False,
        positive_evidence_promotion_performed=False,
    )
    atomic_write_json(
        output / "acquisition_report.json",
        report,
    )

    print()
    print("Generic corpus acquisition M3 complete")
    print("Profile:", args.profile_id)
    print("Selected works:", total)
    print(
        "Access:",
        f"direct_pdf={report.access_resolved_direct_pdf_count}",
        f"landing_only={report.access_resolved_landing_only_count}",
        f"unresolved={report.access_unresolved_count}",
    )
    print(
        "Artifacts:",
        f"downloaded={report.artifact_downloaded_count}",
        f"failed={report.artifact_download_failed_count}",
        f"not_attempted={report.artifact_not_attempted_count}",
    )
    print(
        "Download diagnostics:",
        f"location_attempts={report.total_download_location_attempts}",
        f"multi_location_recoveries={report.multi_location_recovery_count}",
    )
    print(
        "Failure reasons:",
        report.download_failure_reason_counts,
    )
    print(
        "Upstream provider-query coverage:",
        f"{successful_upstream}/{upstream_total}",
        f"warning={report.upstream_coverage_warning}",
    )
    print(
        "Paywall bypass attempted:",
        report.paywall_bypass_attempted,
    )
    print(
        "Positive-evidence promotion:",
        report.positive_evidence_promotion_performed,
    )
    print(
        "Supplementary discovery:",
        report.supplementary_discovery,
    )
    print("Report:", output / "acquisition_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
