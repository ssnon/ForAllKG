from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pipeline_core.bridge_domain import BridgeDomainAdapter
from pipeline_core.bridge_schemas import BridgeChunkGraph


def bridge_output_path(
    chunk_id: str,
    output_dir: str | Path,
) -> Path:
    safe_chunk_id = chunk_id.replace(
        ":",
        "__",
    )
    return (
        Path(output_dir)
        / f"{safe_chunk_id}.json"
    )

def bridge_candidates_path(
    chunk_id: str,
    output_dir: str | Path,
) -> Path:
    safe_chunk_id = chunk_id.replace(
        ":",
        "__",
    )

    return (
        Path(output_dir)
        / f"{safe_chunk_id}__candidates.json"
    )

def bridge_rejections_path(
    chunk_id: str,
    output_dir: str | Path,
) -> Path:
    safe_chunk_id = chunk_id.replace(
        ":",
        "__",
    )
    return (
        Path(output_dir)
        / f"{safe_chunk_id}__rejections.json"
    )

def bridge_relation_repairs_path(
    chunk_id: str,
    output_dir: str | Path,
) -> Path:
    safe_chunk_id = chunk_id.replace(
        ":",
        "__",
    )
    return (
        Path(output_dir)
        / (
            f"{safe_chunk_id}"
            "__relation_repairs.json"
        )
    )

def bridge_candidate_issues_path(
    chunk_id: str,
    output_dir: str | Path,
) -> Path:
    safe_chunk_id = chunk_id.replace(
        ":",
        "__",
    )

    return (
        Path(output_dir)
        / (
            f"{safe_chunk_id}"
            "__candidate_issues.json"
        )
    )

def filter_bridge_raw_chunk(
    *,
    raw_result: BridgeChunkGraph,
    strict_nodes: list[dict[str, Any]],
    chunk_id: str,
    source_payload: dict[str, Any],
    output_dir: str | Path,
    bridge_adapter: BridgeDomainAdapter,
    relation_repairer: Callable[
        [BridgeChunkGraph],
        tuple[BridgeChunkGraph, list[Any]],
    ],
) -> dict[str, Any]:
    """
    Apply deterministic repairs and the current
    Bridge policy to one raw Bridge extraction.

    This function performs no LLM calls.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = bridge_output_path(
        chunk_id,
        output_dir,
    )
    rejection_path = (
        bridge_rejections_path(
            chunk_id,
            output_dir,
        )
    )
    repair_path = (
        bridge_relation_repairs_path(
            chunk_id,
            output_dir,
        )
    )
    candidate_path = (
        bridge_candidates_path(
            chunk_id,
            output_dir,
        )
    )

    candidate_issues_path = (
        bridge_candidate_issues_path(
            chunk_id,
            output_dir,
        )
    )

    bridge_adapter.validate_chunk(
        raw_result,
        paper_id=str(
            source_payload["paper_id"]
        ),
        chunk_id=str(
            source_payload["chunk_id"]
        ),
        document_id=str(
            source_payload["document_id"]
        ),
        document_role=str(
            source_payload["document_role"]
        ),
        page_ids=list(
            source_payload.get(
                "page_ids",
                [],
            )
        ),
        asset_ids=list(
            source_payload.get(
                "asset_ids",
                [],
            )
        ),
        core_text=str(
            source_payload["core_text"]
        ),
        strict_nodes=strict_nodes,
    )

    repaired_result, relation_repairs = (
        relation_repairer(
            raw_result
        )
    )

    partition = bridge_adapter.partition_result(
        repaired_result,
        strict_nodes=strict_nodes,
        core_text=str(
            source_payload["core_text"]
        ),
    )

    filtered_result = (
        partition.accepted
    )
    candidate_result = (
        partition.candidates
    )
    candidate_records = list(
        partition.candidate_records
    )
    rejections = list(
        partition.rejections
    )

    output_path.write_text(
        json.dumps(
            filtered_result.model_dump(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rejection_path.write_text(
        json.dumps(
            [
                rejection.to_dict()
                for rejection in rejections
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    repair_path.write_text(
        json.dumps(
            [
                repair.to_dict()
                for repair
                in relation_repairs
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    candidate_path.write_text(
        json.dumps(
            candidate_result.model_dump(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    candidate_issues_path.write_text(
        json.dumps(
            [
                record.to_dict()
                for record
                in candidate_records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "paper_id": (
            filtered_result.paper_id
        ),
        "chunk_id": (
            filtered_result.chunk_id
        ),
        "document_id": (
            filtered_result.document_id
        ),
        "document_role": (
            filtered_result.document_role
        ),
        "section": (
            filtered_result.section
        ),
        "output_path": str(
            output_path
        ),
        "rejections_path": str(
            rejection_path
        ),
        "relation_repairs_path": str(
            repair_path
        ),
        "concept_count": len(
            filtered_result.concepts
        ),
        "pattern_count": sum(
            concept.retention_lane
            == "accepted_pattern"
            for concept
            in filtered_result.concepts
        ),
        "frontier_count": sum(
            concept.retention_lane
            == "paper_local_frontier"
            for concept
            in filtered_result.concepts
        ),
        "link_count": len(
            filtered_result.links
        ),
        "rejection_count": len(
            rejections
        ),
        "relation_repair_count": len(
            relation_repairs
        ),
        "candidates_path": str(
            candidate_path
        ),
        "candidate_issues_path": str(
            candidate_issues_path
        ),
        "candidate_count": len(
            candidate_result.concepts
        ),
        "fatal_rejection_count": len(
            rejections
        ),
    }
