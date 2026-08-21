from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline_core.corpus.extraction.document_config import load_paper_configs
from pipeline_core.literature.acquisition.access_contracts import CorpusSourceAcquisitionReport, SourceArtifact
from pipeline_core.literature.acquisition.contracts import CorpusSelectionReport, SelectedCorpusWork
from pipeline_core.literature.acquisition.materialization_contracts import CorpusMaterializationReport, PaperMaterializationRecord
from pipeline_core.literature.acquisition.materialization_package import (
    generated_paper_config_entry,
    materialize_artifact,
    stable_paper_id,
    write_extraction_plan,
    write_generated_config,
)
from scripts.literature.materialization_plan_runtime import EXTRACT_PAPER_COMMAND_PREFIX
from pipeline_core.literature.acquisition.materialization_policy import load_materialization_policy
from pipeline_core.literature.acquisition.materialization_state import atomic_write_json, load_state, state_matches_sources, state_path, write_jsonl, write_state
from pipeline_core.literature.acquisition.progress import compact_text, progress_prefix
from pipeline_core.literature.acquisition.supplementary_contracts import SupplementaryAcquisitionReport
from pipeline_core.literature.catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path, model) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows


def _artifact_sort_key(artifact: SourceArtifact):
    return (
        artifact.role != "main",
        artifact.artifact_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic Corpus Acquisition M4: materialize downloaded main/SI "
            "source artifacts into existing GraphAgentsDAC DocumentPackage "
            "layout, emit a generated v3 papers config and extraction plan, "
            "without running LLM extraction."
        )
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--domain-profile-id", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--selected-works", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--m3-dir", required=True, type=Path)
    parser.add_argument(
        "--m3-1-dir",
        type=Path,
        default=None,
        help=(
            "Optional completed M3.1 directory. If supplied, downloaded SI "
            "artifacts are materialized and included in generated config."
        ),
    )
    parser.add_argument("--materialization-policy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-config", required=True, type=Path)
    parser.add_argument("--materialization-id", required=True)
    parser.add_argument("--paper-id-prefix", required=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-materialize paper states containing failed documents.",
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
            f"Completed M3 report required: {m3_report_path}"
        )
    m3_report = CorpusSourceAcquisitionReport.model_validate_json(
        m3_report_path.read_text(encoding="utf-8")
    )
    main_artifacts = _read_jsonl(
        args.m3_dir / "artifacts.jsonl",
        SourceArtifact,
    )

    m31_report_path = None
    supplementary_artifacts: list[SourceArtifact] = []
    if args.m3_1_dir is not None:
        m31_report_path = (
            args.m3_1_dir
            / "supplementary_acquisition_report.json"
        )
        if not m31_report_path.exists():
            raise FileNotFoundError(
                "When --m3-1-dir is supplied it must be complete: "
                f"{m31_report_path}"
            )
        m31_report = SupplementaryAcquisitionReport.model_validate_json(
            m31_report_path.read_text(encoding="utf-8")
        )
        if m31_report.source_catalog_id != packet.catalog_id:
            raise ValueError("M3.1/catalog mismatch")
        if m31_report.source_profile_id != args.profile_id:
            raise ValueError("M3.1/profile mismatch")
        supplementary_artifacts = _read_jsonl(
            args.m3_1_dir / "supplementary_artifacts.jsonl",
            SourceArtifact,
        )

    policy = load_materialization_policy(
        args.materialization_policy
    )

    if packet.acquisition_profile_id != args.profile_id:
        raise ValueError("Catalog/profile mismatch")
    if selection_report.profile_id != args.profile_id:
        raise ValueError("Selection/profile mismatch")
    if selection_report.source_catalog_id != packet.catalog_id:
        raise ValueError("Selection/catalog mismatch")
    if m3_report.source_profile_id != args.profile_id:
        raise ValueError("M3/profile mismatch")
    if m3_report.source_catalog_id != packet.catalog_id:
        raise ValueError("M3/catalog mismatch")

    selected_ids = [row.work_id for row in selected]
    if selected_ids != selection_report.selected_work_ids:
        raise ValueError(
            "selected_works.jsonl does not match selection_report.json"
        )

    work_map: dict[str, CatalogWork] = {
        row.work_id: row for row in packet.works
    }
    artifacts_by_work: dict[str, list[SourceArtifact]] = defaultdict(list)
    for artifact in [*main_artifacts, *supplementary_artifacts]:
        if artifact.work_id in set(selected_ids):
            artifacts_by_work[artifact.work_id].append(artifact)

    output_root = args.output_dir.resolve()
    package_root = output_root / "packages"
    state_root = output_root / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    all_documents = []
    paper_records: list[PaperMaterializationRecord] = []
    generated_papers: dict[str, dict[str, Any]] = {}
    paper_map_rows = []
    total = len(selected)

    for index, selected_row in enumerate(selected, start=1):
        work = work_map[selected_row.work_id]
        paper_id = stable_paper_id(
            prefix=args.paper_id_prefix,
            work_id=work.work_id,
        )
        source_artifacts = sorted(
            artifacts_by_work.get(work.work_id, []),
            key=_artifact_sort_key,
        )
        downloaded = [
            row for row in source_artifacts
            if row.status == "downloaded"
        ]

        main_sources = [
            row for row in downloaded if row.role == "main"
        ]
        si_sources = [
            row for row in downloaded
            if row.role == "supporting_information"
        ]

        print(
            progress_prefix("M4", index, total),
            f"paper={paper_id}",
            f"main={len(main_sources)}",
            f"si={len(si_sources)}",
            compact_text(work.title, max_length=52),
            flush=True,
        )

        state_file = state_path(state_root, paper_id)
        prior = load_state(state_file)
        prior_failed = bool(
            prior
            and any(
                row.status == "failed"
                for row in prior
            )
        )
        if (
            prior is not None
            and state_matches_sources(
                path=state_file,
                artifacts=downloaded,
            )
            and not (args.retry_failed and prior_failed)
        ):
            documents = prior
            print(
                progress_prefix("M4", index, total),
                "resume",
                " ".join(
                    f"{row.document_id}:{row.status}"
                    for row in documents
                ),
                flush=True,
            )
        else:
            documents = []
            if len(main_sources) > 1:
                raise RuntimeError(
                    f"Multiple downloaded main artifacts for {work.work_id}"
                )

            if main_sources:
                main_package = (
                    package_root
                    / paper_id
                    / "main"
                    / "main"
                )
                documents.append(
                    materialize_artifact(
                        materialization_id=args.materialization_id,
                        paper_id=paper_id,
                        work=work,
                        document_id="main",
                        role="main",
                        artifact=main_sources[0],
                        package_dir=main_package,
                        policy=policy,
                        project_root=PROJECT_ROOT,
                    )
                )
            else:
                # No fake DocumentPackage is created without main source.
                placeholder_artifact = SourceArtifact(
                    artifact_id=f"missing_main:{work.work_id}",
                    work_id=work.work_id,
                    role="main",
                    status="not_attempted",
                )
                documents.append(
                    materialize_artifact(
                        materialization_id=args.materialization_id,
                        paper_id=paper_id,
                        work=work,
                        document_id="main",
                        role="main",
                        artifact=placeholder_artifact,
                        package_dir=(
                            package_root / paper_id / "main" / "main"
                        ),
                        policy=policy,
                        project_root=PROJECT_ROOT,
                    )
                )

            for si_index, artifact in enumerate(
                sorted(si_sources, key=lambda row: row.artifact_id),
                start=1,
            ):
                document_id = f"si{si_index}"
                package_dir = (
                    package_root
                    / paper_id
                    / document_id
                    / f"si_{si_index}"
                )
                documents.append(
                    materialize_artifact(
                        materialization_id=args.materialization_id,
                        paper_id=paper_id,
                        work=work,
                        document_id=document_id,
                        role="supporting_information",
                        artifact=artifact,
                        package_dir=package_dir,
                        policy=policy,
                        project_root=PROJECT_ROOT,
                    )
                )

            write_state(
                path=state_file,
                documents=documents,
                source_artifact_ids=[
                    row.artifact_id for row in downloaded
                ],
                source_artifact_sha256={
                    row.artifact_id: row.sha256
                    for row in downloaded
                },
            )
            print(
                progress_prefix("M4", index, total),
                " ".join(
                    f"{row.document_id}:{row.status}"
                    for row in documents
                ),
                flush=True,
            )

        all_documents.extend(documents)
        main_row = next(
            row for row in documents if row.role == "main"
        )
        si_rows = [
            row for row in documents
            if row.role == "supporting_information"
        ]
        config_entry = generated_paper_config_entry(
            paper_id=paper_id,
            documents=documents,
            policy=policy,
        )
        extraction_ready = config_entry is not None
        if config_entry is not None:
            generated_papers[paper_id] = config_entry

        paper_records.append(
            PaperMaterializationRecord(
                paper_id=paper_id,
                work_id=work.work_id,
                title=work.title,
                doi=work.doi,
                main_document_status=main_row.status,
                supplementary_document_count=len(si_rows),
                supplementary_materialized_count=sum(
                    row.status == "materialized"
                    for row in si_rows
                ),
                extraction_ready=extraction_ready,
                document_ids=[
                    row.document_id for row in documents
                    if row.status == "materialized"
                ],
            )
        )
        paper_map_rows.append(
            {
                "paper_id": paper_id,
                "work_id": work.work_id,
                "title": work.title,
                "doi": work.doi,
                "extraction_ready": extraction_ready,
            }
        )

    args.generated_config.parent.mkdir(parents=True, exist_ok=True)
    write_generated_config(
        config_path=args.generated_config,
        papers=generated_papers,
    )

    # Prove the generated file is accepted by the existing config parser.
    parsed = load_paper_configs(
        args.generated_config,
        project_root=PROJECT_ROOT,
    )
    if set(parsed) != set(generated_papers):
        raise RuntimeError("Generated config parser roundtrip mismatch")

    extraction_plan_path = output_root / "extraction_plan.jsonl"
    write_extraction_plan(
        path=extraction_plan_path,
        paper_ids=sorted(generated_papers),
        generated_config_path=args.generated_config,
        domain_profile_id=args.domain_profile_id,
        data_root=args.data_root,
        extract_command_prefix=EXTRACT_PAPER_COMMAND_PREFIX,
        project_root=PROJECT_ROOT,
    )
    write_jsonl(
        output_root / "materialized_documents.jsonl",
        all_documents,
    )
    write_jsonl(
        output_root / "paper_materialization_records.jsonl",
        paper_records,
    )
    write_jsonl(
        output_root / "paper_map.jsonl",
        paper_map_rows,
    )

    status_counts = Counter(row.status for row in all_documents)
    main_downloaded = sum(
        row.status == "downloaded"
        for row in main_artifacts
        if row.role == "main"
    )
    supplementary_downloaded = sum(
        row.status == "downloaded"
        for row in supplementary_artifacts
        if row.role == "supporting_information"
    )
    main_materialized = sum(
        row.role == "main" and row.status == "materialized"
        for row in all_documents
    )
    main_failed = sum(
        row.role == "main" and row.status == "failed"
        for row in all_documents
    )
    si_materialized = sum(
        row.role == "supporting_information"
        and row.status == "materialized"
        for row in all_documents
    )
    si_failed = sum(
        row.role == "supporting_information"
        and row.status == "failed"
        for row in all_documents
    )
    ready_count = sum(row.extraction_ready for row in paper_records)

    report = CorpusMaterializationReport(
        materialization_id=args.materialization_id,
        source_profile_id=args.profile_id,
        source_catalog_id=packet.catalog_id,
        source_m3_report_path=str(m3_report_path),
        source_m3_1_report_path=(
            str(m31_report_path)
            if m31_report_path is not None
            else None
        ),
        policy_id=policy.policy_id,
        selected_work_count=total,
        main_downloaded_source_count=main_downloaded,
        main_materialized_count=main_materialized,
        main_materialization_failed_count=main_failed,
        supplementary_downloaded_source_count=supplementary_downloaded,
        supplementary_materialized_count=si_materialized,
        supplementary_materialization_failed_count=si_failed,
        unsupported_source_count=status_counts["unsupported"],
        extraction_ready_paper_count=ready_count,
        not_extraction_ready_paper_count=total - ready_count,
        generated_config_path=str(args.generated_config),
        extraction_plan_path=str(extraction_plan_path),
        output_root=str(output_root),
        llm_calls_performed=False,
        scientific_result_inference_performed=False,
        positive_evidence_promotion_performed=False,
    )
    atomic_write_json(
        output_root / "materialization_report.json",
        report,
    )

    print()
    print("Generic corpus acquisition M4 complete")
    print("Selected works:", total)
    print(
        "Main:",
        f"downloaded_source={main_downloaded}",
        f"materialized={main_materialized}",
        f"failed={main_failed}",
    )
    print(
        "SI:",
        f"downloaded_source={supplementary_downloaded}",
        f"materialized={si_materialized}",
        f"failed={si_failed}",
        f"unsupported={status_counts['unsupported']}",
    )
    print(
        "Extraction ready:",
        f"{ready_count}/{total}",
    )
    print("Generated config:", args.generated_config)
    print("Extraction plan:", extraction_plan_path)
    print("LLM calls performed:", report.llm_calls_performed)
    print(
        "Positive-evidence promotion:",
        report.positive_evidence_promotion_performed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
