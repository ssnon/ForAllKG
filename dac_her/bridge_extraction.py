from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dac_her.bridge_policy import filter_bridge_result
from dac_her.bridge_prompts import BRIDGE_SYSTEM_PROMPT, build_bridge_prompt
from dac_her.bridge_schemas import BridgeChunkGraph
from dac_her.bridge_validation import validate_bridge_chunk
from dac_her.graph_io import knowledge_graph_to_networkx
from dac_her.scientific_signatures import strict_node_catalog
from dac_her.schemas import KnowledgeGraph
from dac_her.bridge_repair import (
    repair_rejected_bridge_candidates,
)
from dac_her.bridge_relation_repairs import (
    apply_deterministic_relation_repairs,
)

def bridge_output_path(chunk_id: str, output_dir: str | Path) -> Path:
    safe_chunk_id = chunk_id.replace(":", "__")
    return Path(output_dir) / f"{safe_chunk_id}.json"


def bridge_rejections_path(chunk_id: str, output_dir: str | Path) -> Path:
    safe_chunk_id = chunk_id.replace(":", "__")
    return Path(output_dir) / f"{safe_chunk_id}__rejections.json"

def bridge_raw_output_path(
    chunk_id: str,
    output_dir: str | Path,
) -> Path:
    safe_chunk_id = chunk_id.replace(
        ":",
        "__",
    )
    return (
        Path(output_dir)
        / f"{safe_chunk_id}__raw.json"
    )

def _catalog(result: KnowledgeGraph) -> list[dict[str, Any]]:
    return strict_node_catalog(knowledge_graph_to_networkx(result))


def _load_cached(
    *,
    output_path: Path,
    strict_result: KnowledgeGraph,
    source_payload: dict[str, Any],
) -> BridgeChunkGraph | None:
    if not output_path.exists():
        return None
    try:
        result = BridgeChunkGraph.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        nodes = _catalog(strict_result)
        validate_bridge_chunk(
            result,
            paper_id=str(source_payload["paper_id"]),
            chunk_id=str(source_payload["chunk_id"]),
            document_id=str(source_payload["document_id"]),
            document_role=str(source_payload["document_role"]),
            page_ids=list(source_payload.get("page_ids", [])),
            asset_ids=list(source_payload.get("asset_ids", [])),
            core_text=str(source_payload["core_text"]),
            strict_nodes=nodes,
        )
        filtered, _ = filter_bridge_result(
            result,
            strict_nodes=nodes,
            core_text=str(source_payload["core_text"]),
        )
        if filtered.model_dump() != result.model_dump():
            return None
        return result
    except Exception:
        return None


def extract_bridge_chunk(
    *,
    strict_result: KnowledgeGraph,
    source_payload: dict[str, Any],
    model: str,
    provider: str | None,
    output_dir: str | Path,
    debug_dir: str | Path,
    force: bool = False,
    max_repairs: int = 2,
    max_tokens: int = 3400,
) -> dict[str, Any]:
    output_path = bridge_output_path(strict_result.chunk_id, output_dir)
    rejection_path = bridge_rejections_path(strict_result.chunk_id, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = bridge_raw_output_path(strict_result.chunk_id,output_dir,)

    if not force:
        cached = _load_cached(
            output_path=output_path,
            strict_result=strict_result,
            source_payload=source_payload,
        )
        if cached is not None:
            rejection_count = 0
            if rejection_path.exists():
                try:
                    rejection_payload = json.loads(
                        rejection_path.read_text(encoding="utf-8")
                    )
                    rejection_count = len(rejection_payload)
                except Exception:
                    rejection_count = 0
            return {
                "status": "skipped",
                "paper_id": cached.paper_id,
                "chunk_id": cached.chunk_id,
                "document_id": cached.document_id,
                "document_role": cached.document_role,
                "section": cached.section,
                "output_path": str(output_path),
                "rejections_path": str(rejection_path),
                "concept_count": len(cached.concepts),
                "pattern_count": sum(
                    concept.retention_lane == "accepted_pattern"
                    for concept in cached.concepts
                ),
                "frontier_count": sum(
                    concept.retention_lane == "paper_local_frontier"
                    for concept in cached.concepts
                ),
                "link_count": len(cached.links),
                "rejection_count": rejection_count,
            }

    # Delay OpenRouter import until a cache miss actually requires an LLM call.
    from dac_her.llm_openrouter import OpenRouterLLM

    nodes = _catalog(strict_result)
    validation_feedback: str | None = None
    attempts: list[dict[str, Any]] = []

    for attempt in range(max_repairs + 1):
        llm = OpenRouterLLM(
            model=model,
            provider=provider,
            reproducible=False,
            zdr=True,
        )
        safe_id = strict_result.chunk_id.replace(":", "__")
        debug_path = debug_dir / f"{safe_id}__attempt_{attempt}.json"

        try:
            raw_result = llm.generate_structured(
                system_prompt=BRIDGE_SYSTEM_PROMPT,
                prompt=build_bridge_prompt(
                    paper_id=str(source_payload["paper_id"]),
                    chunk_id=str(source_payload["chunk_id"]),
                    document_id=str(source_payload["document_id"]),
                    document_role=str(source_payload["document_role"]),
                    section=str(source_payload["section"]),
                    page_ids=list(source_payload.get("page_ids", [])),
                    asset_ids=list(source_payload.get("asset_ids", [])),
                    strict_nodes=nodes,
                    core_text=str(source_payload["core_text"]),
                    validation_feedback=validation_feedback,
                ),
                response_model=BridgeChunkGraph,
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort="minimal",
                debug_path=debug_path,
            )

            validate_bridge_chunk(
                raw_result,
                paper_id=str(source_payload["paper_id"]),
                chunk_id=str(source_payload["chunk_id"]),
                document_id=str(source_payload["document_id"]),
                document_role=str(source_payload["document_role"]),
                page_ids=list(source_payload.get("page_ids", [])),
                asset_ids=list(source_payload.get("asset_ids", [])),
                core_text=str(source_payload["core_text"]),
                strict_nodes=nodes,
            )
            raw_output_path.write_text(
                json.dumps(
                    raw_result.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            repaired_result, relation_repairs = (
                apply_deterministic_relation_repairs(
                    raw_result
                )
            )

            result, rejections = filter_bridge_result(
                repaired_result,
                strict_nodes=nodes,
                core_text=str(
                    source_payload["core_text"]
                ),
            )
            repair_log_path = (
                output_path.parent
                / (
                    output_path.stem
                    + "__relation_repairs.json"
                )
            )

            repair_log_path.write_text(
                json.dumps(
                    [
                        repair.to_dict()
                        for repair in relation_repairs
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            result, rejections = (
                filter_bridge_result(
                    raw_result,
                    strict_nodes=nodes,
                    core_text=str(
                        source_payload["core_text"]
                    ),
                )
            )

            metadata = dict(llm.last_call_metadata or {})
            attempts.append(metadata)
            output_path.write_text(
                json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            rejection_path.write_text(
                json.dumps(
                    [rejection.to_dict() for rejection in rejections],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {
                "status": "success",
                "paper_id": result.paper_id,
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "document_role": result.document_role,
                "section": result.section,
                "output_path": str(output_path),
                "rejections_path": str(rejection_path),
                "concept_count": len(result.concepts),
                "pattern_count": sum(
                    concept.retention_lane == "accepted_pattern"
                    for concept in result.concepts
                ),
                "frontier_count": sum(
                    concept.retention_lane == "paper_local_frontier"
                    for concept in result.concepts
                ),
                "link_count": len(result.links),
                "rejection_count": len(rejections),
                "semantic_repairs": attempt,
                "attempts": attempts,
                "raw_output_path": str(raw_output_path),
                "relation_repairs_path": str(
                    repair_log_path
                ),
                "relation_repair_count": len(
                    relation_repairs
                ),
                **metadata,
            }
        except (ValidationError, ValueError) as error:
            metadata = dict(llm.last_call_metadata or {})
            attempts.append(metadata)
            if attempt < max_repairs:
                validation_feedback = str(error)
                continue
            return {
                "status": "failed",
                "paper_id": strict_result.paper_id,
                "chunk_id": strict_result.chunk_id,
                "document_id": strict_result.document_id,
                "document_role": strict_result.document_role,
                "section": strict_result.section,
                "output_path": str(output_path),
                "rejections_path": str(rejection_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "attempts": attempts,
                **metadata,
            }
        except Exception as error:
            metadata = dict(llm.last_call_metadata or {})
            attempts.append(metadata)
            return {
                "status": "failed",
                "paper_id": strict_result.paper_id,
                "chunk_id": strict_result.chunk_id,
                "document_id": strict_result.document_id,
                "document_role": strict_result.document_role,
                "section": strict_result.section,
                "output_path": str(output_path),
                "rejections_path": str(rejection_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "attempts": attempts,
                **metadata,
            }

    raise RuntimeError("Unexpected bridge extraction loop exit.")
