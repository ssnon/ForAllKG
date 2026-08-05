from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dac_her.bridge_prompts import BRIDGE_SYSTEM_PROMPT, build_bridge_prompt
from dac_her.bridge_schemas import BridgeChunkGraph
from dac_her.bridge_validation import validate_bridge_chunk
from dac_her.graph_io import knowledge_graph_to_networkx
from dac_her.scientific_signatures import strict_node_catalog
from dac_her.schemas import KnowledgeGraph

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


def _load_cached_raw(
    *,
    raw_output_path: Path,
    strict_result: KnowledgeGraph,
    source_payload: dict[str, Any],
) -> BridgeChunkGraph | None:
    """
    Load a cached raw LLM extraction.

    Raw cache validity depends only on:
    - BridgeChunkGraph schema validity
    - source/chunk metadata consistency
    - strict-node anchor validity
    - source grounding validity

    It must not depend on the current Bridge policy.
    """
    if not raw_output_path.exists():
        return None

    try:
        raw_result = (
            BridgeChunkGraph.model_validate_json(
                raw_output_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        validate_bridge_chunk(
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
            strict_nodes=_catalog(
                strict_result
            ),
        )

        return raw_result

    except Exception:
        return None

def extract_bridge_raw_chunk(
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
    """
    Extract and persist one raw BridgeChunkGraph.

    This function performs:
    - raw-cache loading
    - LLM extraction on cache miss
    - structural/source validation
    - raw JSON persistence

    It does not perform:
    - deterministic relation repair
    - policy filtering
    - rejection generation
    - Bridge graph assembly
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_output_path = (
        bridge_raw_output_path(
            strict_result.chunk_id,
            output_dir,
        )
    )

    if not force:
        cached = _load_cached_raw(
            raw_output_path=raw_output_path,
            strict_result=strict_result,
            source_payload=source_payload,
        )

        if cached is not None:
            return {
                "status": "skipped",
                "paper_id": cached.paper_id,
                "chunk_id": cached.chunk_id,
                "document_id": (
                    cached.document_id
                ),
                "document_role": (
                    cached.document_role
                ),
                "section": cached.section,
                "raw_output_path": str(
                    raw_output_path
                ),
                "raw_concept_count": len(
                    cached.concepts
                ),
                "raw_pattern_count": sum(
                    concept.retention_lane
                    == "accepted_pattern"
                    for concept
                    in cached.concepts
                ),
                "raw_frontier_count": sum(
                    concept.retention_lane
                    == "paper_local_frontier"
                    for concept
                    in cached.concepts
                ),
                "raw_link_count": len(
                    cached.links
                ),
            }

    # Import OpenRouter only when a raw cache miss
    # actually requires an LLM call.
    from dac_her.llm_openrouter import (
        OpenRouterLLM,
    )

    nodes = _catalog(strict_result)
    validation_feedback: str | None = None
    attempts: list[dict[str, Any]] = []

    for attempt in range(
        max_repairs + 1
    ):
        llm = OpenRouterLLM(
            model=model,
            provider=provider,
            reproducible=False,
            zdr=True,
        )

        safe_id = (
            strict_result.chunk_id.replace(
                ":",
                "__",
            )
        )
        debug_path = (
            debug_dir
            / (
                f"{safe_id}"
                f"__attempt_{attempt}.json"
            )
        )

        try:
            raw_result = (
                llm.generate_structured(
                    system_prompt=(
                        BRIDGE_SYSTEM_PROMPT
                    ),
                    prompt=build_bridge_prompt(
                        paper_id=str(
                            source_payload[
                                "paper_id"
                            ]
                        ),
                        chunk_id=str(
                            source_payload[
                                "chunk_id"
                            ]
                        ),
                        document_id=str(
                            source_payload[
                                "document_id"
                            ]
                        ),
                        document_role=str(
                            source_payload[
                                "document_role"
                            ]
                        ),
                        section=str(
                            source_payload[
                                "section"
                            ]
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
                        strict_nodes=nodes,
                        core_text=str(
                            source_payload[
                                "core_text"
                            ]
                        ),
                        validation_feedback=(
                            validation_feedback
                        ),
                    ),
                    response_model=(
                        BridgeChunkGraph
                    ),
                    temperature=0.0,
                    max_tokens=max_tokens,
                    reasoning_effort=(
                        "minimal"
                    ),
                    debug_path=debug_path,
                )
            )

            validate_bridge_chunk(
                raw_result,
                paper_id=str(
                    source_payload["paper_id"]
                ),
                chunk_id=str(
                    source_payload["chunk_id"]
                ),
                document_id=str(
                    source_payload[
                        "document_id"
                    ]
                ),
                document_role=str(
                    source_payload[
                        "document_role"
                    ]
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

            metadata = dict(
                llm.last_call_metadata
                or {}
            )
            attempts.append(metadata)

            return {
                "status": "success",
                "paper_id": (
                    raw_result.paper_id
                ),
                "chunk_id": (
                    raw_result.chunk_id
                ),
                "document_id": (
                    raw_result.document_id
                ),
                "document_role": (
                    raw_result.document_role
                ),
                "section": (
                    raw_result.section
                ),
                "raw_output_path": str(
                    raw_output_path
                ),
                "raw_concept_count": len(
                    raw_result.concepts
                ),
                "raw_pattern_count": sum(
                    concept.retention_lane
                    == "accepted_pattern"
                    for concept
                    in raw_result.concepts
                ),
                "raw_frontier_count": sum(
                    concept.retention_lane
                    == "paper_local_frontier"
                    for concept
                    in raw_result.concepts
                ),
                "raw_link_count": len(
                    raw_result.links
                ),
                "validation_repairs": (
                    attempt
                ),
                "attempts": attempts,
                **metadata,
            }

        except (
            ValidationError,
            ValueError,
        ) as error:
            metadata = dict(
                llm.last_call_metadata
                or {}
            )
            attempts.append(metadata)

            if attempt < max_repairs:
                validation_feedback = str(
                    error
                )
                continue

            return {
                "status": "failed",
                "paper_id": (
                    strict_result.paper_id
                ),
                "chunk_id": (
                    strict_result.chunk_id
                ),
                "document_id": (
                    strict_result.document_id
                ),
                "document_role": (
                    strict_result.document_role
                ),
                "section": (
                    strict_result.section
                ),
                "raw_output_path": str(
                    raw_output_path
                ),
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(error),
                "attempts": attempts,
                **metadata,
            }

        except Exception as error:
            metadata = dict(
                llm.last_call_metadata
                or {}
            )
            attempts.append(metadata)

            return {
                "status": "failed",
                "paper_id": (
                    strict_result.paper_id
                ),
                "chunk_id": (
                    strict_result.chunk_id
                ),
                "document_id": (
                    strict_result.document_id
                ),
                "document_role": (
                    strict_result.document_role
                ),
                "section": (
                    strict_result.section
                ),
                "raw_output_path": str(
                    raw_output_path
                ),
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(error),
                "attempts": attempts,
                **metadata,
            }

    raise RuntimeError(
        "Unexpected raw Bridge "
        "extraction loop exit."
    )


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
    """
    Temporary backward-compatible wrapper.

    Existing scripts may continue calling
    extract_bridge_chunk while extraction and policy
    run orchestration are being separated.
    """
    raw_record = extract_bridge_raw_chunk(
        strict_result=strict_result,
        source_payload=source_payload,
        model=model,
        provider=provider,
        output_dir=output_dir,
        debug_dir=debug_dir,
        force=force,
        max_repairs=max_repairs,
        max_tokens=max_tokens,
    )

    if raw_record["status"] == "failed":
        return raw_record

    raw_output_path = Path(
        str(
            raw_record[
                "raw_output_path"
            ]
        )
    )

    raw_result = (
        BridgeChunkGraph.model_validate_json(
            raw_output_path.read_text(
                encoding="utf-8"
            )
        )
    )
    
    from dac_her.bridge_filtering import (
        filter_bridge_raw_chunk,
    )
    filtered_record = (
        filter_bridge_raw_chunk(
            raw_result=raw_result,
            strict_result=strict_result,
            source_payload=source_payload,
            output_dir=output_dir,
        )
    )

    return {
        **filtered_record,
        "status": "success",
        "raw_status": (
            raw_record["status"]
        ),
        "raw_output_path": str(
            raw_output_path
        ),
        "attempts": raw_record.get(
            "attempts",
            [],
        ),
        "validation_repairs": (
            raw_record.get(
                "validation_repairs",
                0,
            )
        ),
    }