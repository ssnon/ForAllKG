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
from dac_her.corpus_acquisition.contracts import (
    CorpusSelectionReport,
    SelectedCorpusWork,
)
from dac_her.corpus_acquisition.progress import (
    compact_text,
    progress_prefix,
)
from dac_her.corpus_acquisition.supplementary_acquisition import (
    SupplementaryArtifactDownloader,
)
from dac_her.corpus_acquisition.supplementary_contracts import (
    SupplementaryAcquisitionReport,
)
from dac_her.corpus_acquisition.supplementary_policy import (
    load_supplementary_discovery_policy,
)
from dac_her.corpus_acquisition.supplementary_resolution import (
    SupplementaryArtifactResolver,
)
from dac_her.corpus_acquisition.supplementary_state import (
    access_resolution_sha256,
    atomic_write_json,
    load_state,
    safe_state_name,
    state_matches_main_access,
    write_jsonl,
    write_state,
)
from dac_her.literature_catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


def _read_jsonl(path: Path, model) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic Corpus Acquisition M3.1: discover supplementary "
            "research artifacts from explicit Crossref relationships and "
            "public landing-page links, then conservatively acquire verified "
            "high-confidence direct files. No publisher-specific URL guessing, "
            "authentication, paywall bypass, or positive-evidence promotion."
        )
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--selected-works", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--m3-dir", required=True, type=Path)
    parser.add_argument("--supplementary-policy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--acquisition-id", required=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry work states containing failed supplementary downloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = LiteratureCatalogPacket.model_validate_json(
        args.catalog.read_text(encoding="utf-8")
    )
    selected = _read_jsonl(args.selected_works, SelectedCorpusWork)
    selection_report = CorpusSelectionReport.model_validate_json(
        args.selection_report.read_text(encoding="utf-8")
    )
    m3_report_path = args.m3_dir / "acquisition_report.json"
    if not m3_report_path.exists():
        raise FileNotFoundError(
            "M3 acquisition_report.json is required; wait for M3 to finish: "
            f"{m3_report_path}"
        )
    m3_report = CorpusSourceAcquisitionReport.model_validate_json(
        m3_report_path.read_text(encoding="utf-8")
    )
    m3_resolutions = _read_jsonl(
        args.m3_dir / "access_resolutions.jsonl",
        AccessResolution,
    )
    policy = load_supplementary_discovery_policy(
        args.supplementary_policy
    )

    if packet.acquisition_profile_id != args.profile_id:
        raise ValueError("Catalog/profile mismatch")
    if selection_report.profile_id != args.profile_id:
        raise ValueError("Selection/profile mismatch")
    if m3_report.source_profile_id != args.profile_id:
        raise ValueError("M3/profile mismatch")
    if m3_report.source_catalog_id != packet.catalog_id:
        raise ValueError("M3/catalog mismatch")

    selected_ids = [row.work_id for row in selected]
    if selected_ids != selection_report.selected_work_ids:
        raise ValueError(
            "selected_works.jsonl does not match selection_report.json"
        )
    resolution_map = {row.work_id: row for row in m3_resolutions}
    if len(resolution_map) != len(m3_resolutions):
        raise ValueError("Duplicate M3 access resolution work_id")

    work_map: dict[str, CatalogWork] = {
        row.work_id: row for row in packet.works
    }
    missing = [
        work_id
        for work_id in selected_ids
        if work_id not in work_map
    ]
    if missing:
        raise ValueError(
            f"Selected work missing from catalog: {missing[:10]!r}"
        )

    output = args.output_dir
    state_root = output / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    resolver = SupplementaryArtifactResolver(policy)
    downloader = SupplementaryArtifactDownloader(policy)

    discoveries = []
    artifacts: list[SourceArtifact] = []
    total = len(selected)

    for index, selected_row in enumerate(selected, start=1):
        work = work_map[selected_row.work_id]
        state_path = state_root / safe_state_name(work.work_id)
        prior = load_state(state_path)
        current_access = resolution_map.get(work.work_id)
        current_access_sha = (
            access_resolution_sha256(current_access)
            if current_access is not None
            else None
        )
        prior_failed = bool(
            prior
            and any(
                row.status == "download_failed"
                for row in prior[1]
            )
        )
        upstream_access_unchanged = (
            prior is not None
            and state_matches_main_access(
                path=state_path,
                main_access_sha256=current_access_sha,
            )
        )
        if (
            prior is not None
            and upstream_access_unchanged
            and not (args.retry_failed and prior_failed)
        ):
            discovery, work_artifacts = prior
            print(
                progress_prefix("M3.1", index, total),
                "resume",
                f"status={discovery.status}",
                f"candidates={len(discovery.candidates)}",
                f"artifacts={len(work_artifacts)}",
                compact_text(work.title, max_length=56),
                flush=True,
            )
        else:
            print(
                progress_prefix("M3.1", index, total),
                "discover",
                f"doi={work.doi or '-'}",
                compact_text(work.title, max_length=68),
                flush=True,
            )
            discovery = resolver.discover(
                work=work,
                main_access=current_access,
            )
            print(
                progress_prefix("M3.1", index, total),
                f"status={discovery.status}",
                f"candidates={len(discovery.candidates)}",
                flush=True,
            )

            work_artifacts = []
            for candidate in discovery.candidates:
                artifact = downloader.acquire(
                    candidate=candidate,
                    output_root=output,
                )
                work_artifacts.append(artifact)
            counts = Counter(row.status for row in work_artifacts)
            print(
                progress_prefix("M3.1", index, total),
                (
                    "supp="
                    f"downloaded:{counts['downloaded']} "
                    f"failed:{counts['download_failed']} "
                    f"not_attempted:{counts['not_attempted']}"
                ),
                flush=True,
            )
            write_state(
                path=state_path,
                discovery=discovery,
                artifacts=work_artifacts,
                main_access_sha256=current_access_sha,
            )

        discoveries.append(discovery)
        artifacts.extend(work_artifacts)

    write_jsonl(output / "supplementary_discoveries.jsonl", discoveries)
    write_jsonl(output / "supplementary_artifacts.jsonl", artifacts)

    discovery_counts = Counter(row.status for row in discoveries)
    confidence_counts = Counter(
        candidate.confidence
        for row in discoveries
        for candidate in row.candidates
    )
    artifact_counts = Counter(row.status for row in artifacts)
    resolver_attempts = [
        attempt
        for row in discoveries
        for attempt in row.resolver_attempts
    ]

    report = SupplementaryAcquisitionReport(
        acquisition_id=args.acquisition_id,
        source_profile_id=args.profile_id,
        source_catalog_id=packet.catalog_id,
        source_m3_report_path=str(m3_report_path),
        policy_id=policy.policy_id,
        selected_work_count=total,
        direct_file_candidate_work_count=discovery_counts[
            "direct_file_candidates"
        ],
        metadata_only_candidate_work_count=discovery_counts[
            "metadata_only_candidates"
        ],
        unresolved_work_count=discovery_counts["unresolved"],
        candidate_count=sum(
            len(row.candidates) for row in discoveries
        ),
        high_confidence_candidate_count=confidence_counts["high"],
        medium_confidence_candidate_count=confidence_counts["medium"],
        low_confidence_candidate_count=confidence_counts["low"],
        supplementary_artifact_downloaded_count=artifact_counts["downloaded"],
        supplementary_artifact_download_failed_count=artifact_counts[
            "download_failed"
        ],
        supplementary_artifact_not_attempted_count=artifact_counts[
            "not_attempted"
        ],
        crossref_relation_attempt_count=sum(
            attempt.resolver == "crossref_relation"
            and attempt.status != "skipped"
            for attempt in resolver_attempts
        ),
        public_landing_attempt_count=sum(
            attempt.resolver == "public_landing_html"
            and attempt.status != "skipped"
            for attempt in resolver_attempts
        ),
        output_root=str(output),
        publisher_specific_url_guessing_performed=False,
        paywall_bypass_attempted=False,
        positive_evidence_promotion_performed=False,
    )
    atomic_write_json(
        output / "supplementary_acquisition_report.json",
        report,
    )

    print()
    print("Generic corpus acquisition M3.1 complete")
    print("Profile:", args.profile_id)
    print("Selected works:", total)
    print(
        "Discovery:",
        f"direct_file_work={report.direct_file_candidate_work_count}",
        f"metadata_only_work={report.metadata_only_candidate_work_count}",
        f"unresolved={report.unresolved_work_count}",
    )
    print(
        "Candidates:",
        f"total={report.candidate_count}",
        f"high={report.high_confidence_candidate_count}",
        f"medium={report.medium_confidence_candidate_count}",
        f"low={report.low_confidence_candidate_count}",
    )
    print(
        "Supplementary artifacts:",
        f"downloaded={report.supplementary_artifact_downloaded_count}",
        f"failed={report.supplementary_artifact_download_failed_count}",
        f"not_attempted={report.supplementary_artifact_not_attempted_count}",
    )
    print(
        "Publisher-specific URL guessing:",
        report.publisher_specific_url_guessing_performed,
    )
    print("Paywall bypass attempted:", report.paywall_bypass_attempted)
    print(
        "Positive-evidence promotion:",
        report.positive_evidence_promotion_performed,
    )
    print(
        "Report:",
        output / "supplementary_acquisition_report.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
