from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import dac_her.chunking as chunking_module
import dac_her.schemas as schemas_module
import dac_her.extraction as extraction_module
import dac_her.graph_normalization as graph_normalization_module
import dac_her.llm_openrouter as llm_openrouter_module
import dac_her.measurement_scalarization as measurement_scalarization_module
import dac_her.structural_repair as structural_repair_module
import dac_her.validation as validation_module
import dac_her.chunking_recovery as chunking_recovery_module
import dac_her.draft_schema as draft_schema_module
import dac_her.graph_validation as graph_validation_module
import dac_her.extraction_quality as extraction_quality_module
import dac_her.lossless_normalization as lossless_normalization_module
import dac_her.recovery_policy as recovery_policy_module
import dac_her.semantic_patch as semantic_patch_module
import dac_her.semantic_patch_prompts as semantic_patch_prompts_module
import dac_her.semantic_patch_schema as semantic_patch_schema_module
import dac_her.strict_recovery as strict_recovery_module
import dac_her.strict_validation as strict_validation_module
import dac_her.validation_issues as validation_issues_module
import dac_her.micro_reextract_prompts as micro_reextract_prompts_module

from dac_her.asset_index import AssetRecord, assets_by_id, write_assets_jsonl
from dac_her.chunking import ChunkSpec, count_tokens, create_chunks, split_chunk_in_half
from dac_her.chunking_recovery import split_chunk_structurally
from dac_her.config import DocumentConfig, get_paper_config
from dac_her.document_package import (
    DocumentPackage,
    extract_supplementary_references,
    load_document_package,
    select_document_sources,
)
from dac_her.extraction import chunk_output_path, extract_one_chunk, load_existing_result
from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.registry import get_domain_profile
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.extraction_quality import (
    QUALITY_PARTIAL_CRITICAL,
    QUALITY_REJECTED,
    annotate_quarantined_records,
    evaluate_extraction_quality,
)
from dac_her.figure_extraction import (
    FigureAnalysis,
    analyze_figure,
    format_asset_context,
    resolve_vision_model,
    should_analyze_asset,
)
from dac_her.run_state import (
    compute_run_metadata,
    run_directory,
    write_json,
    write_latest_run_pointer,
)
from dac_her.vocab_registry import load_default_registries


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract configured main/SI documents, preserving Marker assets "
            "and optional figure-vision provenance."
        )
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--domain-profile", default="dac_her")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--provider",
        default=os.getenv("OPENROUTER_PROVIDER") or None,
    )
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore valid chunk caches in the current fingerprinted run.",
    )
    parser.add_argument(
        "--import-legacy-cache",
        action="store_true",
        help=(
            "Import only matching schema-valid JSONs from the legacy "
            "<paper>/chunks directory. New document-aware chunk IDs mean "
            "pre-v2 caches will normally be rejected."
        ),
    )
    parser.add_argument(
        "--vision-all",
        action="store_true",
        help="Analyze every linked image asset with the vision model.",
    )
    parser.add_argument(
        "--vision-asset",
        action="append",
        default=[],
        help=(
            "Analyze one asset ID, relative path, or basename. Repeat for "
            "multiple assets."
        ),
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Override OPENROUTER_VISION_MODEL and per-document vision_model.",
    )
    parser.add_argument(
        "--force-vision",
        action="store_true",
        help="Regenerate cached figure analyses.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Legacy/explicit override for PARTIAL_CRITICAL extraction. "
            "PARTIAL_ACCEPTABLE now exits successfully by default; REJECTED "
            "runs never become usable through this flag."
        ),
    )
    return parser.parse_args()


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_source_chunk(
    source_chunk_dir: Path,
    chunk: ChunkSpec,
) -> Path:
    """Persist the exact source leaf used by strict and bridge extraction."""
    source_chunk_dir.mkdir(parents=True, exist_ok=True)
    safe_chunk_id = chunk.chunk_id.replace(":", "__")
    path = source_chunk_dir / f"{safe_chunk_id}.json"
    payload = {
        "paper_id": chunk.paper_id,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_role": chunk.document_role,
        "section": chunk.section,
        "chunk_index": chunk.index,
        "split_depth": chunk.split_depth,
        "page_ids": list(chunk.page_ids),
        "asset_ids": list(chunk.asset_ids),
        "asset_paths": list(chunk.asset_paths),
        "asset_locators": list(chunk.asset_locators),
        "left_context": chunk.left_context,
        "core_text": chunk.core_text,
        "right_context": chunk.right_context,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def record_active_chunk(
    active: dict[str, dict[str, Any]],
    record: dict[str, Any],
    *,
    chunk: ChunkSpec,
    source_chunk_dir: Path,
) -> None:
    source_path_value = record.get(
        "source_path"
    )

    source_path = (
        Path(source_path_value)
        if source_path_value
        else write_source_chunk(
            source_chunk_dir,
            chunk,
        )
    )
    active[str(record["chunk_id"])] = {
        "paper_id": record["paper_id"],
        "chunk_id": record["chunk_id"],
        "document_id": record["document_id"],
        "document_role": record["document_role"],
        "section": record["section"],
        "page_ids": record.get("page_ids", []),
        "asset_ids": record.get("asset_ids", []),
        "chunk_index": record["chunk_index"],
        "split_depth": record["split_depth"],
        "status": record["status"],
        "output_path": record["output_path"],
        "source_path": str(source_path),
        "source_tokens_estimated": record.get("source_tokens_estimated"),
        "node_count": record.get("node_count"),
        "edge_count": record.get("edge_count"),
        "unregistered_vocabulary_count": record.get(
            "unregistered_vocabulary_count", 0
        ),
        "vocabulary_issues": record.get("vocabulary_issues", []),
        "acceptance_mode": record.get("acceptance_mode"),
        "normalization_count": record.get("normalization_count", 0),
        "normalization_path": record.get("normalization_path"),
        "patch_attempts": record.get("patch_attempts", 0),
        "micro_reextract_attempts": record.get("micro_reextract_attempts", 0),
        "post_micro_patch_attempts": record.get("post_micro_patch_attempts", 0),
        "patch_operation_count": record.get("patch_operation_count", 0),
        "patch_paths": record.get("patch_paths", []),
        "attempt_usages": record.get("attempt_usages", []),
        "validation_issue_counts": record.get("validation_issue_counts", {}),
    }


def _document_manifest(
    config: DocumentConfig,
    package: DocumentPackage,
) -> dict[str, Any]:
    return {
        "paper_id": package.paper_id,
        "document_id": package.document_id,
        "role": package.role,
        "package_dir": str(package.package_dir),
        "markdown_path": str(package.markdown_path),
        "metadata_path": (
            str(package.metadata_path) if package.metadata_path else None
        ),
        "selection": asdict(config.selection),
        "figure_processing": asdict(config.figure_processing),
        "asset_count": len(package.assets),
        "missing_asset_count": sum(not asset.exists for asset in package.assets),
    }


def main() -> None:
    args = parse_args()
    domain_profile = get_domain_profile(args.domain_profile)
    extraction_adapter = get_extraction_adapter(domain_profile.profile_id)
    data_root = args.data_root or extraction_adapter.default_data_root
    model = args.model or os.getenv("OPENROUTER_EXTRACT_MODEL")
    if not model:
        raise RuntimeError(
            "OPENROUTER_EXTRACT_MODEL is not defined and --model was not supplied."
        )

    paper = get_paper_config(
        args.config,
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
    )
    for document in paper.documents:
        if not document.markdown_path.exists():
            raise FileNotFoundError(
                f"Markdown not found for {document.document_id}: "
                f"{document.markdown_path}"
            )

    experiment_registry, metric_registry = load_default_registries(
        PROJECT_ROOT
    )
    vocabulary_context = "\n".join([
        "REGISTERED EXPERIMENT METHODS:",
        *experiment_registry.prompt_lines(metadata_keys=("family",)),
        "",
        "REGISTERED MEASUREMENT METRICS:",
        *metric_registry.prompt_lines(metadata_keys=("canonical_unit", "parameters")),
    ])

    policy = ExtractionPolicy()
    if args.concurrency is not None:
        if args.concurrency < 1:
            raise ValueError("--concurrency must be at least 1.")
        policy = replace(policy, concurrency=args.concurrency)

    run_metadata = compute_run_metadata(
        project_root=PROJECT_ROOT,
        paper=paper,
        policy=policy,
        model=model,
        provider=args.provider,
        schemas_path=schemas_module.__file__,
        chunking_path=chunking_module.__file__,
        runtime_options={
            "vision_all": args.vision_all,
            "vision_assets": sorted(args.vision_asset),
            "vision_model_override": args.vision_model,
            "domain_profile_id": domain_profile.profile_id,
            "extraction_adapter_id": extraction_adapter.adapter_id,
            "data_root": str(data_root),
        },
        prompt_version=extraction_adapter.prompt_version,
        system_prompt=extraction_adapter.system_prompt,
        domain_profile_id=domain_profile.profile_id,
        data_root=data_root,
        implementation_paths=(
            extraction_module.__file__,
            graph_normalization_module.__file__,
            llm_openrouter_module.__file__,
            measurement_scalarization_module.__file__,
            structural_repair_module.__file__,
            validation_module.__file__,
            chunking_recovery_module.__file__,
            draft_schema_module.__file__,
            graph_validation_module.__file__,
            extraction_quality_module.__file__,
            lossless_normalization_module.__file__,
            recovery_policy_module.__file__,
            semantic_patch_module.__file__,
            semantic_patch_prompts_module.__file__,
            semantic_patch_schema_module.__file__,
            strict_recovery_module.__file__,
            strict_validation_module.__file__,
            validation_issues_module.__file__,
            micro_reextract_prompts_module.__file__,
        ),
    )
    run_id = str(run_metadata["run_id"])
    run_dir = run_directory(
        PROJECT_ROOT, paper.paper_id, run_id, data_root=data_root
    )
    chunk_output_dir = run_dir / "chunks"
    source_chunk_dir = run_dir / "source_chunks"
    debug_dir = run_dir / "debug"
    documents_dir = run_dir / "documents"
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_dir = run_dir / "attempts" / attempt_id
    manifest_path = attempt_dir / "manifest.jsonl"
    events_path = run_dir / "events.jsonl"

    for directory in (
        run_dir,
        chunk_output_dir,
        source_chunk_dir,
        debug_dir,
        documents_dir,
        attempt_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.touch(exist_ok=False)
    write_json(run_dir / "run.json", run_metadata)
    write_latest_run_pointer(
        project_root=PROJECT_ROOT,
        paper_id=paper.paper_id,
        run_metadata=run_metadata,
        data_root=data_root,
    )

    configs_by_id = {document.document_id: document for document in paper.documents}
    packages: dict[str, DocumentPackage] = {
        document.document_id: load_document_package(
            paper_id=paper.paper_id,
            config=document,
        )
        for document in paper.documents
    }

    all_assets: dict[str, AssetRecord] = {}
    document_records: list[dict[str, Any]] = []
    for document in paper.documents:
        package = packages[document.document_id]
        document_dir = documents_dir / document.document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        document_record = _document_manifest(document, package)
        document_records.append(document_record)
        write_json(document_dir / "document.json", document_record)
        write_assets_jsonl(document_dir / "assets.jsonl", package.assets)
        for asset in package.assets:
            if asset.asset_id in all_assets:
                raise ValueError(f"Duplicate asset ID: {asset.asset_id}")
            all_assets[asset.asset_id] = asset

    write_json(run_dir / "documents.json", {"documents": document_records})
    write_json(
        run_dir / "asset_manifest.json",
        {"assets": [asset.to_dict() for asset in all_assets.values()]},
    )

    selected_sources = []
    main_source_texts: list[str] = []

    # Main and independently selected documents are processed first so their
    # Supplementary references can drive SI referenced_blocks selection.
    for document in paper.documents:
        if (
            document.role == "supporting_information"
            and document.selection.mode == "referenced_blocks"
        ):
            continue
        sources = select_document_sources(
            package=packages[document.document_id],
            config=document,
        )
        selected_sources.extend(sources)
        if document.role == "main":
            main_source_texts.extend(source.text for source in sources)

    si_reference_documents = [
        document
        for document in paper.documents
        if document.role == "supporting_information"
        and document.selection.mode == "referenced_blocks"
    ]
    use_whole_main_for_references = any(
        document.selection.reference_scope == "whole_main"
        for document in si_reference_documents
    )
    reference_texts = (
        [
            packages[document.document_id].markdown
            for document in paper.documents
            if document.role == "main"
        ]
        if use_whole_main_for_references
        else main_source_texts
    )
    supplementary_references = extract_supplementary_references(reference_texts)

    si_selection_diagnostics: list[dict[str, Any]] = []
    for document in si_reference_documents:
        sources = select_document_sources(
            package=packages[document.document_id],
            config=document,
            supplementary_references=supplementary_references,
        )
        selected_sources.extend(sources)
        si_selection_diagnostics.append({
            "document_id": document.document_id,
            "document_role": document.role,
            "reference_scope": document.selection.reference_scope,
            "fallback": document.selection.fallback,
            "references_detected": list(supplementary_references),
            "selected_block_count": len(sources),
            "selected_sections": [source.section for source in sources],
        })

    write_json(
        run_dir / "supplementary_references.json",
        {
            "reference_scope": (
                "whole_main" if use_whole_main_for_references
                else "selected_main"
            ),
            "references": list(supplementary_references),
        },
    )
    write_json(
        run_dir / "si_selection_diagnostics.json",
        {"documents": si_selection_diagnostics},
    )

    initial_chunks: list[ChunkSpec] = []
    source_summaries: list[dict[str, Any]] = []
    for source in selected_sources:
        package = packages[source.document_id]
        chunks = create_chunks(
            paper_id=paper.paper_id,
            document_id=source.document_id,
            document_role=source.document_role,
            section=source.section,
            section_text=source.text,
            policy=policy,
            assets=package.assets,
        )
        if not chunks:
            raise RuntimeError(
                f"Chunking produced no chunks for "
                f"{source.document_id}/{source.section}"
            )
        initial_chunks.extend(chunks)
        source_summaries.append({
            "document_id": source.document_id,
            "document_role": source.document_role,
            "selection_id": source.selection_id,
            "section": source.section,
            "source_tokens_estimated": count_tokens(source.text),
            "initial_chunks": len(chunks),
        })

    linked_asset_ids = {
        asset_id for chunk in initial_chunks for asset_id in chunk.asset_ids
    }
    requested_assets = tuple(args.vision_asset)
    analyses: dict[str, FigureAnalysis] = {}

    for asset_id in sorted(linked_asset_ids):
        asset = all_assets[asset_id]
        document_config = configs_by_id[asset.document_id]
        if not should_analyze_asset(
            asset,
            document_config.figure_processing,
            force_all=args.vision_all,
            requested_assets=requested_assets,
        ):
            continue

        vision_model = args.vision_model or resolve_vision_model(
            document_config.figure_processing,
            model,
        )
        analysis = analyze_figure(
            asset=asset,
            model=vision_model,
            provider=args.provider,
            output_dir=documents_dir / asset.document_id / "vision",
            force=args.force_vision,
        )
        analyses[asset_id] = analysis
        print("[VISION]", asset_id, flush=True)

    def decorate_chunk(chunk: ChunkSpec) -> ChunkSpec:
        assets = [all_assets[asset_id] for asset_id in chunk.asset_ids]
        return replace(
            chunk,
            asset_context=format_asset_context(assets, analyses),
        )

    initial_chunks = [decorate_chunk(chunk) for chunk in initial_chunks]

    if args.import_legacy_cache:
        legacy_dir = (
            Path(data_root) / "extracted" / paper.paper_id / "chunks"
        )
        imported = 0
        if legacy_dir.exists():
            for chunk in initial_chunks:
                legacy_path = chunk_output_path(chunk, legacy_dir)
                current_path = chunk_output_path(chunk, chunk_output_dir)
                if current_path.exists() or not legacy_path.exists():
                    continue
                if load_existing_result(chunk=chunk, output_path=legacy_path) is None:
                    print("[LEGACY CACHE REJECTED]", legacy_path, flush=True)
                    continue
                shutil.copy2(legacy_path, current_path)
                imported += 1
                print("[LEGACY CACHE IMPORTED]", chunk.chunk_id, flush=True)
        print("Legacy chunks imported:", imported, flush=True)

    queue: deque[ChunkSpec] = deque(initial_chunks)
    active_chunks: dict[str, dict[str, Any]] = {}
    failed_records: list[dict[str, Any]] = []
    quarantined_records: list[dict[str, Any]] = []
    lineage_records: list[dict[str, Any]] = []
    success_count = skipped_count = failed_count = split_count = 0
    quarantine_count = semantic_rechunk_count = 0

    print("Model:", model, flush=True)
    print("Domain profile:", domain_profile.profile_id, flush=True)
    print("Extraction adapter:", extraction_adapter.adapter_id, flush=True)
    print("Data root:", data_root, flush=True)
    print("Paper ID:", paper.paper_id, flush=True)
    print("Run ID:", run_id, flush=True)
    print("Documents:", len(paper.documents), flush=True)
    print("Selected sources:", len(selected_sources), flush=True)
    print("Indexed assets:", len(all_assets), flush=True)
    print("Linked assets:", len(linked_asset_ids), flush=True)
    print("Vision analyses:", len(analyses), flush=True)
    print("Initial chunks:", len(initial_chunks), flush=True)
    print(
        "Estimated source tokens:",
        sum(count_tokens(chunk.core_text) for chunk in initial_chunks),
        flush=True,
    )

    while queue:
        logical_batch: list[ChunkSpec] = []
        while queue and len(logical_batch) < policy.logical_batch_size:
            logical_batch.append(queue.popleft())
        print("\nStarting logical batch:", len(logical_batch), "chunks", flush=True)

        with ThreadPoolExecutor(max_workers=policy.concurrency) as executor:
            future_map = {
                executor.submit(
                    extract_one_chunk,
                    chunk=chunk,
                    model=model,
                    provider=args.provider,
                    policy=policy,
                    chunk_output_dir=chunk_output_dir,
                    debug_dir=debug_dir,
                    experiment_registry=experiment_registry,
                    metric_registry=metric_registry,
                    vocabulary_context=vocabulary_context,
                    force=(args.force or args.force_vision),
                    extraction_adapter=extraction_adapter,
                ): chunk
                for chunk in logical_batch
            }

            for future in as_completed(future_map):
                chunk = future_map[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "status": "failed",
                        "paper_id": chunk.paper_id,
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_role": chunk.document_role,
                        "section": chunk.section,
                        "page_ids": list(chunk.page_ids),
                        "asset_ids": list(chunk.asset_ids),
                        "chunk_index": chunk.index,
                        "split_depth": chunk.split_depth,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }

                source_path = write_source_chunk(
                    source_chunk_dir,
                    chunk,
                )

                record.setdefault(
                    "source_path",
                    str(source_path),
                )
                record["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
                append_manifest(manifest_path, record)
                append_manifest(events_path, {"attempt_id": attempt_id, **record})
                status = record["status"]

                if status == "success":
                    success_count += 1
                    record_active_chunk(
                        active_chunks,
                        record,
                        chunk=chunk,
                        source_chunk_dir=source_chunk_dir,
                    )
                    utilization = record.get("utilization")
                    print(
                        "[SUCCESS]",
                        chunk.chunk_id,
                        f"document={chunk.document_id}",
                        "source_tokens=",
                        record["source_tokens_estimated"],
                        "output_tokens=",
                        record.get("output_tokens", "unknown"),
                        "utilization=",
                        f"{utilization:.2%}" if utilization is not None else "unknown",
                        "nodes=",
                        record.get("node_count", "unknown"),
                        "edges=",
                        record.get("edge_count", "unknown"),
                        flush=True,
                    )
                elif status == "skipped":
                    skipped_count += 1
                    record_active_chunk(
                        active_chunks,
                        record,
                        chunk=chunk,
                        source_chunk_dir=source_chunk_dir,
                    )
                    print("[SKIPPED]", chunk.chunk_id, flush=True)
                elif status in {"truncated", "rechunk"}:
                    if chunk.split_depth >= policy.max_split_depth:
                        terminal_record = dict(record)
                        terminal_record["status"] = "quarantined"
                        terminal_record["error_type"] = "MaxSplitDepthExceeded"
                        terminal_record["error_message"] = (
                            "Recovery requested another split after reaching "
                            f"max_split_depth={policy.max_split_depth}."
                        )
                        quarantine_count += 1
                        quarantined_records.append(terminal_record)
                        print("[QUARANTINED: MAX SPLIT DEPTH]", chunk.chunk_id, flush=True)
                        continue

                    try:
                        if status == "rechunk":
                            rechunk_result = split_chunk_structurally(
                                chunk,
                                policy,
                                reason=record.get("recovery_reason", "semantic_failure"),
                            )
                            children = [decorate_chunk(child) for child in rechunk_result.children]
                            lineage_records.append({
                                "parent_chunk_id": chunk.chunk_id,
                                "child_chunk_ids": [child.chunk_id for child in children],
                                "split_method": rechunk_result.split_method,
                                "reason": rechunk_result.reason,
                            })
                            semantic_rechunk_count += 1
                        else:
                            children = [
                                decorate_chunk(child)
                                for child in split_chunk_in_half(chunk, policy)
                            ]
                            lineage_records.append({
                                "parent_chunk_id": chunk.chunk_id,
                                "child_chunk_ids": [child.chunk_id for child in children],
                                "split_method": "token_midpoint_truncation",
                                "reason": "output_truncation",
                            })
                    except RuntimeError as split_error:
                        terminal_record = dict(record)
                        terminal_record["status"] = "quarantined"
                        terminal_record["error_type"] = "UnsplittableRecovery"
                        terminal_record["error_message"] = str(split_error)
                        quarantine_count += 1
                        quarantined_records.append(terminal_record)
                        print("[QUARANTINED: UNSPLITTABLE]", chunk.chunk_id, split_error, flush=True)
                        continue

                    for child in reversed(children):
                        queue.appendleft(child)
                    split_count += 1
                    print(
                        "[SPLIT]",
                        chunk.chunk_id,
                        "->",
                        [child.chunk_id for child in children],
                        flush=True,
                    )
                elif status == "quarantined":
                    quarantine_count += 1
                    quarantined_records.append(record)
                    print(
                        "[QUARANTINED]",
                        chunk.chunk_id,
                        record.get("recovery_reason", "unknown"),
                        flush=True,
                    )
                else:
                    failed_count += 1
                    failed_records.append(record)
                    print(
                        "[FAILED]",
                        chunk.chunk_id,
                        record.get("error_message", "Unknown error"),
                        flush=True,
                    )

    # Quarantined leaves remain excluded from active_chunks and therefore
    # can never enter the canonical graph. We classify paper-level usability
    # separately from strict completeness.
    quarantined_records = annotate_quarantined_records(
        quarantined_records
    )
    quality = evaluate_extraction_quality(
        active_records=active_chunks.values(),
        quarantined_records=quarantined_records,
        failed_records=failed_records,
        policy=policy,
    )

    complete = bool(quality["strict_complete"])
    materialization_status = str(
        quality["graph_materialization_status"]
    )
    paper_status = (
        "complete"
        if complete
        else ("partial" if active_chunks else "quarantined")
    )
    active_payload = {
        "paper_id": paper.paper_id,
        "run_id": run_id,
        "run_fingerprint": run_metadata["run_fingerprint"],
        "paper_status": paper_status,
        "complete": complete,
        "graph_materialization_status": materialization_status,
        "quality": quality,
        "active_chunk_count": len(active_chunks),
        "chunks": sorted(
            active_chunks.values(),
            key=lambda item: (
                str(item["document_id"]),
                str(item["section"]),
                int(item["chunk_index"]),
                str(item["chunk_id"]),
            ),
        ),
        "failed_chunks": failed_records,
        "quarantined_chunks": quarantined_records,
    }
    write_json(run_dir / "active_chunks.json", active_payload)
    write_json(run_dir / "extraction_quality.json", quality)
    write_json(run_dir / "lineage.json", {"splits": lineage_records})

    summary = {
        "paper_id": paper.paper_id,
        "run_id": run_id,
        "complete": complete,
        "documents": document_records,
        "selected_sources": source_summaries,
        "supplementary_references": list(supplementary_references),
        "indexed_assets": len(all_assets),
        "linked_assets": len(linked_asset_ids),
        "vision_analyses": len(analyses),
        "initial_chunks": len(initial_chunks),
        "active_leaf_chunks": len(active_chunks),
        "successful": success_count,
        "skipped": skipped_count,
        "split_operations": split_count,
        "semantic_rechunk_operations": semantic_rechunk_count,
        "quarantined": quarantine_count,
        "failed": failed_count,
        "paper_status": paper_status,
        "graph_materialization_status": materialization_status,
        "extraction_quality": quality,
        "attempt_id": attempt_id,
        "experiment_vocab_version": experiment_registry.version,
        "metric_vocab_version": metric_registry.version,
        "unregistered_vocabulary_count": sum(
            int(item.get("unregistered_vocabulary_count", 0))
            for item in active_chunks.values()
        ),
        "chunk_output_dir": str(chunk_output_dir),
        "source_chunk_dir": str(source_chunk_dir),
        "manifest_path": str(manifest_path),
        "active_chunks_path": str(run_dir / "active_chunks.json"),
    }
    write_json(run_dir / "summary.json", summary)

    print("\nExtraction finished", flush=True)
    print("Successful:", success_count, flush=True)
    print("Skipped:", skipped_count, flush=True)
    print("Split operations:", split_count, flush=True)
    print("Quarantined:", quarantine_count, flush=True)
    print("Failed:", failed_count, flush=True)
    print("Paper status:", paper_status, flush=True)
    print("Complete:", complete, flush=True)
    print(
        "Graph materialization status:",
        materialization_status,
        flush=True,
    )
    print(
        "Source-token coverage:",
        quality.get("source_token_coverage"),
        flush=True,
    )
    print(
        "Quarantine tiers:",
        quality.get("quarantine_tier_counts"),
        flush=True,
    )
    print("Run directory:", run_dir, flush=True)
    print("Active chunks:", run_dir / "active_chunks.json", flush=True)

    if materialization_status == QUALITY_REJECTED:
        raise SystemExit(2)

    if (
        materialization_status == QUALITY_PARTIAL_CRITICAL
        and not args.allow_partial
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
