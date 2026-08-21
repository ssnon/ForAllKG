from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline_core.corpus.extraction.document_config import load_paper_configs
from pipeline_core.literature.acquisition.access_contracts import CorpusSourceAcquisitionReport, SourceArtifact
from pipeline_core.literature.acquisition.contracts import CorpusSelectionReport, SelectedCorpusWork
from pipeline_core.literature.acquisition.materialization_contracts import CorpusMaterializationReport, MaterializedDocument, PaperMaterializationRecord
from pipeline_core.literature.acquisition.materialization_package import (
    generated_paper_config_entry,
    materialize_artifact,
    stable_paper_id,
    write_extraction_plan,
    write_generated_config,
)
from scripts.materialization_plan_runtime import EXTRACT_PAPER_COMMAND_PREFIX
from pipeline_core.literature.acquisition.materialization_policy import load_materialization_policy
from pipeline_core.literature.acquisition.materialization_state import atomic_write_json, load_state, state_path, write_jsonl
from pipeline_core.literature.acquisition.progress import compact_text, progress_prefix
from pipeline_core.literature.acquisition.supplementary_contracts import SupplementaryAcquisitionReport
from pipeline_core.literature.catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATE_SCHEMA = "m4-materialization-state-v2"
_CONTEXT_SCHEMA = "m4-materialization-context-v1"
_REPORT_SCHEMA = "m4-incremental-materialization-report-v1"
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class PaperPlan:
    index: int
    total: int
    work: CatalogWork
    paper_id: str
    downloaded: tuple[SourceArtifact, ...]
    main_sources: tuple[SourceArtifact, ...]
    si_sources: tuple[SourceArtifact, ...]


@dataclass(frozen=True)
class PaperResult:
    index: int
    paper_id: str
    documents: tuple[MaterializedDocument, ...]
    cache_actions: dict[str, str]


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_file(path: str | None) -> Path | None:
    if not path:
        return None
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def _implementation_hash(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    return _sha256_file(path) if path.exists() else "missing"


def _materialization_context(*, args, policy) -> dict[str, Any]:
    payload = {
        "schema_version": _CONTEXT_SCHEMA,
        "materialization_id": args.materialization_id,
        "policy_id": policy.policy_id,
        "policy_sha256": _sha256_file(args.materialization_policy),
        "materializers_sha256": _implementation_hash(
            "dac_her/corpus_acquisition/materializers.py"
        ),
        "materialization_package_sha256": _implementation_hash(
            "dac_her/corpus_acquisition/materialization_package.py"
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "fingerprint": _sha256_bytes(canonical),
    }


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _legacy_context_compatible(*, output_root: Path, args, policy) -> bool:
    report = _read_json_mapping(output_root / "materialization_report.json")
    if not report:
        return False
    return (
        report.get("materialization_id") == args.materialization_id
        and report.get("policy_id") == policy.policy_id
    )


def _state_context_compatible(
    *,
    state_file: Path,
    context: dict[str, Any],
    legacy_compatible: bool,
) -> bool:
    payload = _read_json_mapping(state_file)
    if payload is None:
        return False
    prior = payload.get("materialization_context")
    if isinstance(prior, dict) and prior.get("fingerprint"):
        return prior.get("fingerprint") == context.get("fingerprint")
    return legacy_compatible


def _metadata_matches(
    *,
    row: MaterializedDocument,
    artifact: SourceArtifact,
    work: CatalogWork,
) -> bool:
    metadata_path = _project_file(row.metadata_path)
    if metadata_path is None or not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("paper_id") != row.paper_id:
        return False
    if payload.get("work_id") != work.work_id:
        return False
    bibliographic = payload.get("bibliographic") or {}
    if bibliographic.get("title") != work.title:
        return False
    if bibliographic.get("doi") != work.doi:
        return False
    if bibliographic.get("year") != work.year:
        return False
    source = payload.get("source_artifact") or {}
    if source.get("artifact_id") != artifact.artifact_id:
        return False
    if artifact.sha256 and source.get("sha256") != artifact.sha256:
        return False
    return True


def _can_reuse_document(
    *,
    row: MaterializedDocument,
    artifact: SourceArtifact,
    work: CatalogWork,
    materialization_id: str,
    context_compatible: bool,
    retry_failed: bool,
    force: bool,
) -> tuple[bool, str]:
    if force:
        return False, "force"
    if not context_compatible:
        return False, "context_changed"
    if row.materialization_id != materialization_id:
        return False, "materialization_id_changed"
    if row.source_artifact_id != artifact.artifact_id:
        return False, "artifact_id_changed"
    if (
        row.source_artifact_sha256
        and artifact.sha256
        and row.source_artifact_sha256 != artifact.sha256
    ):
        return False, "artifact_sha_changed"
    if row.role != artifact.role:
        return False, "artifact_role_changed"
    if retry_failed and row.status == "failed":
        return False, "retry_failed"

    if row.status == "materialized":
        markdown_path = _project_file(row.markdown_path)
        if markdown_path is None or not markdown_path.exists():
            return False, "markdown_missing"
        if row.markdown_sha256:
            if _sha256_file(markdown_path) != row.markdown_sha256:
                return False, "markdown_sha_changed"
        if not _metadata_matches(row=row, artifact=artifact, work=work):
            return False, "metadata_mismatch"
        return True, "reuse_verified"

    # unsupported/skipped/failed are deterministic terminal records under the
    # same context. Failed records are re-run only when --retry-failed is set.
    return True, f"reuse_{row.status}"


def _prior_by_artifact(
    prior: list[MaterializedDocument] | None,
) -> dict[str, MaterializedDocument]:
    return {
        row.source_artifact_id: row
        for row in (prior or [])
        if row.source_artifact_id
    }


def _assign_si_document_ids(
    *,
    artifacts: tuple[SourceArtifact, ...],
    prior: list[MaterializedDocument] | None,
) -> dict[str, str]:
    prior_map = {
        row.source_artifact_id: row.document_id
        for row in (prior or [])
        if row.role == "supporting_information"
        and row.source_artifact_id
        and row.document_id
    }
    used = set(prior_map.values())
    assigned: dict[str, str] = {}

    numeric_ids = []
    for value in used:
        match = re.fullmatch(r"si(\d+)", value)
        if match:
            numeric_ids.append(int(match.group(1)))
    next_index = max(numeric_ids, default=0) + 1

    for artifact in sorted(artifacts, key=lambda row: row.artifact_id):
        existing = prior_map.get(artifact.artifact_id)
        if existing and existing not in assigned.values():
            assigned[artifact.artifact_id] = existing
            continue
        while f"si{next_index}" in used:
            next_index += 1
        document_id = f"si{next_index}"
        next_index += 1
        used.add(document_id)
        assigned[artifact.artifact_id] = document_id
    return assigned


def _package_dir(
    *,
    package_root: Path,
    paper_id: str,
    document_id: str,
    role: str,
) -> Path:
    if role == "main":
        return package_root / paper_id / "main" / "main"
    match = re.fullmatch(r"si(\d+)", document_id)
    leaf = f"si_{match.group(1)}" if match else document_id
    return package_root / paper_id / document_id / leaf


def _write_state_v2(
    *,
    path: Path,
    documents: list[MaterializedDocument],
    downloaded: tuple[SourceArtifact, ...],
    context: dict[str, Any],
    cache_actions: dict[str, str],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": _STATE_SCHEMA,
            "paper_id": documents[0].paper_id if documents else path.stem,
            "materialization_context": context,
            "source_artifact_ids": [row.artifact_id for row in downloaded],
            "source_artifact_sha256": {
                row.artifact_id: row.sha256 for row in downloaded
            },
            "cache_actions": cache_actions,
            "documents": [row.model_dump(mode="json") for row in documents],
        },
    )


def _materialize_plan(
    *,
    plan: PaperPlan,
    package_root: Path,
    state_root: Path,
    context: dict[str, Any],
    legacy_compatible: bool,
    args,
    policy,
) -> PaperResult:
    with _PRINT_LOCK:
        print(
            progress_prefix("M4i", plan.index, plan.total),
            f"paper={plan.paper_id}",
            f"main={len(plan.main_sources)}",
            f"si={len(plan.si_sources)}",
            compact_text(plan.work.title, max_length=52),
            flush=True,
        )

    if len(plan.main_sources) > 1:
        raise RuntimeError(
            f"Multiple downloaded main artifacts for {plan.work.work_id}"
        )

    state_file = state_path(state_root, plan.paper_id)
    prior = load_state(state_file)
    prior_map = _prior_by_artifact(prior)
    context_compatible = _state_context_compatible(
        state_file=state_file,
        context=context,
        legacy_compatible=legacy_compatible,
    )
    si_ids = _assign_si_document_ids(artifacts=plan.si_sources, prior=prior)

    documents: list[MaterializedDocument] = []
    actions: dict[str, str] = {}

    if plan.main_sources:
        main_artifact = plan.main_sources[0]
    else:
        main_artifact = SourceArtifact(
            artifact_id=f"missing_main:{plan.work.work_id}",
            work_id=plan.work.work_id,
            role="main",
            status="not_attempted",
        )

    jobs: list[tuple[str, str, SourceArtifact]] = [
        ("main", "main", main_artifact)
    ]
    jobs.extend(
        (
            si_ids[artifact.artifact_id],
            "supporting_information",
            artifact,
        )
        for artifact in sorted(plan.si_sources, key=lambda row: row.artifact_id)
    )

    for document_id, role, artifact in jobs:
        cached = prior_map.get(artifact.artifact_id)
        if cached is not None:
            reusable, reason = _can_reuse_document(
                row=cached,
                artifact=artifact,
                work=plan.work,
                materialization_id=args.materialization_id,
                context_compatible=context_compatible,
                retry_failed=args.retry_failed,
                force=args.force,
            )
            if reusable:
                documents.append(cached)
                actions[document_id] = reason
                continue
            actions[document_id] = f"run:{reason}"
        else:
            actions[document_id] = "run:new_source"

        documents.append(
            materialize_artifact(
                materialization_id=args.materialization_id,
                paper_id=plan.paper_id,
                work=plan.work,
                document_id=document_id,
                role=role,
                artifact=artifact,
                package_dir=_package_dir(
                    package_root=package_root,
                    paper_id=plan.paper_id,
                    document_id=document_id,
                    role=role,
                ),
                policy=policy,
                project_root=PROJECT_ROOT,
            )
        )

    _write_state_v2(
        path=state_file,
        documents=documents,
        downloaded=plan.downloaded,
        context=context,
        cache_actions=actions,
    )

    with _PRINT_LOCK:
        reused = sum(value.startswith("reuse_") for value in actions.values())
        executed = len(actions) - reused
        summary = " ".join(
            f"{row.document_id}:{row.status}" for row in documents
        )
        print(
            progress_prefix("M4i", plan.index, plan.total),
            f"cache_reused={reused}",
            f"executed={executed}",
            summary,
            flush=True,
        )

    return PaperResult(
        index=plan.index,
        paper_id=plan.paper_id,
        documents=tuple(documents),
        cache_actions=actions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incremental Generic Corpus Acquisition M4. Reuses verified "
            "document-level materialization outputs and only runs Marker or "
            "other materializers for new/stale documents."
        )
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--domain-profile-id", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--selected-works", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--m3-dir", required=True, type=Path)
    parser.add_argument("--m3-1-dir", type=Path, default=None)
    parser.add_argument("--materialization-policy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-config", required=True, type=Path)
    parser.add_argument("--materialization-id", required=True)
    parser.add_argument("--paper-id-prefix", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel papers to materialize. Default 1. Marker may use GPU/"
            "large RAM; increase conservatively (usually 2 first)."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run only cached documents whose prior status is failed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-materialize all current documents.",
    )
    return parser.parse_args()


def main() -> int:
    started = time.monotonic()
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

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
    main_artifacts = _read_jsonl(args.m3_dir / "artifacts.jsonl", SourceArtifact)

    m31_report_path = None
    supplementary_artifacts: list[SourceArtifact] = []
    if args.m3_1_dir is not None:
        m31_report_path = args.m3_1_dir / "supplementary_acquisition_report.json"
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

    policy = load_materialization_policy(args.materialization_policy)

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
    selected_id_set = set(selected_ids)

    work_map: dict[str, CatalogWork] = {row.work_id: row for row in packet.works}
    missing_catalog = [work_id for work_id in selected_ids if work_id not in work_map]
    if missing_catalog:
        raise ValueError(f"Selected work missing from catalog: {missing_catalog[0]}")

    artifacts_by_work: dict[str, list[SourceArtifact]] = defaultdict(list)
    for artifact in [*main_artifacts, *supplementary_artifacts]:
        if artifact.work_id in selected_id_set:
            artifacts_by_work[artifact.work_id].append(artifact)

    output_root = args.output_dir.resolve()
    package_root = output_root / "packages"
    state_root = output_root / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    context = _materialization_context(args=args, policy=policy)
    legacy_compatible = _legacy_context_compatible(
        output_root=output_root,
        args=args,
        policy=policy,
    )

    plans: list[PaperPlan] = []
    total = len(selected)
    for index, selected_row in enumerate(selected, start=1):
        work = work_map[selected_row.work_id]
        paper_id = stable_paper_id(
            prefix=args.paper_id_prefix,
            work_id=work.work_id,
        )
        downloaded = tuple(
            row
            for row in sorted(
                artifacts_by_work.get(work.work_id, []),
                key=_artifact_sort_key,
            )
            if row.status == "downloaded"
        )
        plans.append(
            PaperPlan(
                index=index,
                total=total,
                work=work,
                paper_id=paper_id,
                downloaded=downloaded,
                main_sources=tuple(row for row in downloaded if row.role == "main"),
                si_sources=tuple(
                    row for row in downloaded
                    if row.role == "supporting_information"
                ),
            )
        )

    results_by_index: dict[int, PaperResult] = {}
    if args.workers == 1:
        for plan in plans:
            result = _materialize_plan(
                plan=plan,
                package_root=package_root,
                state_root=state_root,
                context=context,
                legacy_compatible=legacy_compatible,
                args=args,
                policy=policy,
            )
            results_by_index[result.index] = result
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _materialize_plan,
                    plan=plan,
                    package_root=package_root,
                    state_root=state_root,
                    context=context,
                    legacy_compatible=legacy_compatible,
                    args=args,
                    policy=policy,
                ): plan
                for plan in plans
            }
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results_by_index[result.index] = result
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    all_documents: list[MaterializedDocument] = []
    paper_records: list[PaperMaterializationRecord] = []
    generated_papers: dict[str, dict[str, Any]] = {}
    paper_map_rows = []
    cache_action_counts: Counter[str] = Counter()

    for plan in plans:
        result = results_by_index[plan.index]
        documents = list(result.documents)
        all_documents.extend(documents)
        cache_action_counts.update(result.cache_actions.values())

        main_row = next(row for row in documents if row.role == "main")
        si_rows = [
            row for row in documents
            if row.role == "supporting_information"
        ]
        config_entry = generated_paper_config_entry(
            paper_id=plan.paper_id,
            documents=documents,
            policy=policy,
        )
        extraction_ready = config_entry is not None
        if config_entry is not None:
            generated_papers[plan.paper_id] = config_entry

        paper_records.append(
            PaperMaterializationRecord(
                paper_id=plan.paper_id,
                work_id=plan.work.work_id,
                title=plan.work.title,
                doi=plan.work.doi,
                main_document_status=main_row.status,
                supplementary_document_count=len(si_rows),
                supplementary_materialized_count=sum(
                    row.status == "materialized" for row in si_rows
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
                "paper_id": plan.paper_id,
                "work_id": plan.work.work_id,
                "title": plan.work.title,
                "doi": plan.work.doi,
                "extraction_ready": extraction_ready,
            }
        )

    args.generated_config.parent.mkdir(parents=True, exist_ok=True)
    write_generated_config(config_path=args.generated_config, papers=generated_papers)
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
    write_jsonl(output_root / "materialized_documents.jsonl", all_documents)
    write_jsonl(
        output_root / "paper_materialization_records.jsonl",
        paper_records,
    )
    write_jsonl(output_root / "paper_map.jsonl", paper_map_rows)

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
        row.role == "supporting_information" and row.status == "materialized"
        for row in all_documents
    )
    si_failed = sum(
        row.role == "supporting_information" and row.status == "failed"
        for row in all_documents
    )
    ready_count = sum(row.extraction_ready for row in paper_records)

    report = CorpusMaterializationReport(
        materialization_id=args.materialization_id,
        source_profile_id=args.profile_id,
        source_catalog_id=packet.catalog_id,
        source_m3_report_path=str(m3_report_path),
        source_m3_1_report_path=(
            str(m31_report_path) if m31_report_path is not None else None
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
    atomic_write_json(output_root / "materialization_report.json", report)

    elapsed = time.monotonic() - started
    incremental_report = {
        "schema_version": _REPORT_SCHEMA,
        "materialization_id": args.materialization_id,
        "policy_id": policy.policy_id,
        "source_catalog_id": packet.catalog_id,
        "selected_work_count": total,
        "workers": args.workers,
        "force": args.force,
        "retry_failed": args.retry_failed,
        "legacy_context_compatible": legacy_compatible,
        "materialization_context": context,
        "cache_action_counts": dict(sorted(cache_action_counts.items())),
        "cache_reused_document_count": sum(
            count
            for action, count in cache_action_counts.items()
            if action.startswith("reuse_")
        ),
        "executed_document_count": sum(
            count
            for action, count in cache_action_counts.items()
            if action.startswith("run:")
        ),
        "elapsed_seconds": elapsed,
        "llm_calls_performed": False,
        "scientific_result_inference_performed": False,
        "positive_evidence_promotion_performed": False,
    }
    atomic_write_json(
        output_root / "incremental_materialization_report.json",
        incremental_report,
    )

    print()
    print("Incremental Generic corpus acquisition M4 complete")
    print("Selected works:", total)
    print("Workers:", args.workers)
    print(
        "Cache:",
        f"reused={incremental_report['cache_reused_document_count']}",
        f"executed={incremental_report['executed_document_count']}",
    )
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
    print("Extraction ready:", f"{ready_count}/{total}")
    print("Elapsed seconds:", f"{elapsed:.1f}")
    print("Generated config:", args.generated_config)
    print("Extraction plan:", extraction_plan_path)
    print(
        "Incremental report:",
        output_root / "incremental_materialization_report.json",
    )
    print("LLM calls performed:", False)
    print("Positive-evidence promotion:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
