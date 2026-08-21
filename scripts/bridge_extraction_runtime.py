from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pipeline_core.bridge_domain import BridgeDomainAdapter

from pipeline_core.bridge_draft_schema import (
    BridgeCandidateRepair,
    BridgeChunkDraft,
)
from domains.dac_her.bridge_prompts import (
    BRIDGE_SYSTEM_PROMPT,
    build_bridge_prompt,
)
from domains.dac_her.bridge_recovery import (
    BRIDGE_RECOVERY_VERSION,
    BridgeRecoveryError,
    recover_bridge_draft,
)
from domains.dac_her.bridge_recovery_prompts import (
    BRIDGE_RECOVERY_PROMPT_VERSION,
    BRIDGE_RECOVERY_SYSTEM_PROMPT,
    build_bridge_candidate_repair_prompt,
)
from pipeline_core.bridge_schemas import BridgeChunkGraph
from pipeline_core.bridge_source_reconciliation import (
    BRIDGE_SOURCE_RECONCILIATION_VERSION,
)
from domains.dac_her.bridge_validation import validate_bridge_chunk
from pipeline_core.graph_io import knowledge_graph_to_networkx
from domains.dac_her.scientific_signatures import strict_node_catalog
from pipeline_core.corpus.schemas import KnowledgeGraph


def _resolve_bridge_adapter(
    bridge_adapter: BridgeDomainAdapter | None,
) -> BridgeDomainAdapter:
    if bridge_adapter is not None:
        return bridge_adapter

    # Backward-compatible direct Python callers retain the frozen HER
    # semantics. Domain-aware CLIs resolve explicitly and fail closed.
    from domains.bridge_registry import get_bridge_adapter

    return get_bridge_adapter("dac_her")


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


def _catalog(
    result: KnowledgeGraph,
    bridge_adapter: BridgeDomainAdapter,
) -> list[dict[str, Any]]:
    return bridge_adapter.strict_node_catalog_builder(
        knowledge_graph_to_networkx(result)
    )


def _load_cached_raw(
    *,
    raw_output_path: Path,
    strict_result: KnowledgeGraph,
    source_payload: dict[str, Any],
    bridge_adapter: BridgeDomainAdapter,
) -> BridgeChunkGraph | None:
    """
    Load a cached raw extraction only when it remains strict-valid against the
    current strict graph and source snapshot.
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
            strict_nodes=_catalog(
                strict_result,
                bridge_adapter,
            ),
        )

        return raw_result

    except Exception:
        return None


def _sidecar_paths(
    *,
    raw_output_path: Path,
    chunk_id: str,
) -> tuple[Path, Path]:
    extraction_dir = raw_output_path.parent.parent
    safe_chunk_id = chunk_id.replace(
        ":",
        "__",
    )
    recovery_dir = extraction_dir / "recovery"
    quarantine_dir = extraction_dir / "quarantine"
    recovery_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    quarantine_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return (
        recovery_dir / f"{safe_chunk_id}__recovery.json",
        quarantine_dir / f"{safe_chunk_id}__quarantine.json",
    )


def _read_sidecar_count(
    path: Path,
    key: str,
) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return 0
    value = payload.get(key, 0)
    try:
        return int(value)
    except Exception:
        return 0


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
    max_candidate_repairs_per_chunk: int = 3,
    candidate_repair_max_tokens: int = 1800,
    bridge_adapter: BridgeDomainAdapter | None = None,
) -> dict[str, Any]:
    """
    Extract and persist one strict-valid raw BridgeChunkGraph.

    v2.4 recovery semantics:
    - provider output is first parsed as BridgeChunkDraft, not BridgeChunkGraph;
    - formatting-only source spans are reconciled deterministically;
    - accepted-pattern representation errors are normalized deterministically;
    - hard failures are localized to one candidate/link;
    - a bounded local repair may change grounding but not scientific semantics;
    - unrecoverable candidates are quarantined individually;
    - semantic candidate failures therefore do not fail the whole chunk.

    Whole-chunk failure is reserved for technical/provider/draft-shape/internal
    failures that prevent creation of a strict-valid raw Bridge graph.
    """
    bridge_adapter = _resolve_bridge_adapter(bridge_adapter)

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

    raw_output_path = bridge_raw_output_path(
        strict_result.chunk_id,
        output_dir,
    )
    recovery_path, quarantine_path = (
        _sidecar_paths(
            raw_output_path=raw_output_path,
            chunk_id=strict_result.chunk_id,
        )
    )

    if not force:
        cached = _load_cached_raw(
            raw_output_path=raw_output_path,
            strict_result=strict_result,
            source_payload=source_payload,
            bridge_adapter=bridge_adapter,
        )

        if cached is not None:
            return {
                "status": "skipped",
                "paper_id": cached.paper_id,
                "chunk_id": cached.chunk_id,
                "document_id": cached.document_id,
                "document_role": cached.document_role,
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
                    for concept in cached.concepts
                ),
                "raw_frontier_count": sum(
                    concept.retention_lane
                    == "paper_local_frontier"
                    for concept in cached.concepts
                ),
                "raw_link_count": len(
                    cached.links
                ),
                "recovery_path": (
                    str(recovery_path)
                    if recovery_path.exists()
                    else ""
                ),
                "quarantine_path": (
                    str(quarantine_path)
                    if quarantine_path.exists()
                    else ""
                ),
                "quarantined_candidate_count": (
                    _read_sidecar_count(
                        recovery_path,
                        "quarantined_candidate_count",
                    )
                ),
                "repaired_candidate_count": (
                    _read_sidecar_count(
                        recovery_path,
                        "repaired_candidate_count",
                    )
                ),
            }

    # Import provider client only on a cache miss.
    from pipeline_core.openrouter_llm import OpenRouterLLM

    nodes = _catalog(strict_result, bridge_adapter)
    validation_feedback: str | None = None
    attempts: list[dict[str, Any]] = []

    for attempt in range(max_repairs + 1):
        llm = OpenRouterLLM(
            model=model,
            provider=provider,
            reproducible=False,
            zdr=True,
            application_title="GraphAgents DAC-HER",
            default_debug_path="data_dac/debug/last_invalid_structured_response.json",
        )

        safe_id = strict_result.chunk_id.replace(
            ":",
            "__",
        )
        debug_path = (
            debug_dir
            / f"{safe_id}__attempt_{attempt}.json"
        )

        try:
            draft = llm.generate_structured(
                system_prompt=bridge_adapter.system_prompt,
                prompt=bridge_adapter.build_prompt(
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
                    section=str(
                        source_payload["section"]
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
                        source_payload["core_text"]
                    ),
                    validation_feedback=(
                        validation_feedback
                    ),
                ),
                response_model=BridgeChunkDraft,
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort="minimal",
                debug_path=debug_path,
            )
        except (ValidationError, ValueError) as error:
            metadata = dict(
                llm.last_call_metadata or {}
            )
            metadata.update(
                {
                    "call_kind": "bridge_draft_generation",
                    "generation_attempt": attempt,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            attempts.append(metadata)

            if attempt < max_repairs:
                validation_feedback = str(error)
                continue

            return {
                "status": "failed",
                "failure_class": "technical_draft_generation",
                "paper_id": strict_result.paper_id,
                "chunk_id": strict_result.chunk_id,
                "document_id": strict_result.document_id,
                "document_role": strict_result.document_role,
                "section": strict_result.section,
                "raw_output_path": str(raw_output_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "attempts": attempts,
                **metadata,
            }
        except Exception as error:
            metadata = dict(
                llm.last_call_metadata or {}
            )
            metadata.update(
                {
                    "call_kind": "bridge_draft_generation",
                    "generation_attempt": attempt,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            attempts.append(metadata)
            return {
                "status": "failed",
                "failure_class": "technical_provider_or_runtime",
                "paper_id": strict_result.paper_id,
                "chunk_id": strict_result.chunk_id,
                "document_id": strict_result.document_id,
                "document_role": strict_result.document_role,
                "section": strict_result.section,
                "raw_output_path": str(raw_output_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "attempts": attempts,
                **metadata,
            }

        main_metadata = dict(
            llm.last_call_metadata or {}
        )
        main_metadata.update(
            {
                "call_kind": "bridge_draft_generation",
                "generation_attempt": attempt,
            }
        )
        attempts.append(main_metadata)

        candidate_repair_usages: list[dict[str, Any]] = []

        def repair_callback(
            original_concept: dict[str, Any],
            original_links: list[dict[str, Any]],
            issues: list[str],
            repair_index: int,
        ) -> BridgeCandidateRepair | None:
            concept_id = str(
                original_concept.get(
                    "id",
                    "unknown",
                )
            )
            safe_concept_id = "".join(
                char
                if char.isalnum() or char in "-_"
                else "_"
                for char in concept_id
            )[:100]

            repair_llm = OpenRouterLLM(
                model=model,
                provider=provider,
                reproducible=False,
                zdr=True,
                application_title="GraphAgents DAC-HER",
                default_debug_path="data_dac/debug/last_invalid_structured_response.json",
            )
            repair_debug_path = (
                debug_dir
                / (
                    f"{safe_id}__candidate_"
                    f"{safe_concept_id}__repair_"
                    f"{repair_index}.json"
                )
            )

            source_metadata = {
                "paper_id": str(
                    source_payload["paper_id"]
                ),
                "chunk_id": str(
                    source_payload["chunk_id"]
                ),
                "document_id": str(
                    source_payload["document_id"]
                ),
                "document_role": str(
                    source_payload["document_role"]
                ),
                "section": str(
                    source_payload["section"]
                ),
                "page_ids": list(
                    source_payload.get(
                        "page_ids",
                        [],
                    )
                ),
                "asset_ids": list(
                    source_payload.get(
                        "asset_ids",
                        [],
                    )
                ),
            }

            try:
                repair = repair_llm.generate_structured(
                    system_prompt=(
                        bridge_adapter.recovery_system_prompt
                    ),
                    prompt=(
                        bridge_adapter.build_candidate_repair_prompt(
                            original_concept=(
                                original_concept
                            ),
                            original_links=(
                                original_links
                            ),
                            validation_issues=issues,
                            strict_nodes=nodes,
                            core_text=str(
                                source_payload[
                                    "core_text"
                                ]
                            ),
                            source_metadata=(
                                source_metadata
                            ),
                        )
                    ),
                    response_model=(
                        BridgeCandidateRepair
                    ),
                    temperature=0.0,
                    max_tokens=(
                        candidate_repair_max_tokens
                    ),
                    reasoning_effort="minimal",
                    debug_path=repair_debug_path,
                )
            except Exception as error:
                usage = dict(
                    repair_llm.last_call_metadata
                    or {}
                )
                usage.update(
                    {
                        "call_kind": (
                            "bridge_candidate_repair"
                        ),
                        "candidate_repair_index": (
                            repair_index
                        ),
                        "concept_id": concept_id,
                        "error_type": (
                            type(error).__name__
                        ),
                        "error_message": str(error),
                    }
                )
                candidate_repair_usages.append(
                    usage
                )
                return None

            usage = dict(
                repair_llm.last_call_metadata or {}
            )
            usage.update(
                {
                    "call_kind": (
                        "bridge_candidate_repair"
                    ),
                    "candidate_repair_index": (
                        repair_index
                    ),
                    "concept_id": concept_id,
                }
            )
            candidate_repair_usages.append(usage)
            return repair

        try:
            recovered = recover_bridge_draft(
                draft,
                source_payload=source_payload,
                strict_nodes=nodes,
                repair_callback=repair_callback,
                max_candidate_repairs_per_chunk=(
                    max_candidate_repairs_per_chunk
                ),
                validation_issues_fn=(
                    bridge_adapter.validation_issues
                ),
                validate_chunk_fn=(
                    bridge_adapter.validate_chunk
                ),
            )
        except BridgeRecoveryError as error:
            attempts.extend(candidate_repair_usages)
            return {
                "status": "failed",
                "failure_class": "technical_recovery_invariant",
                "paper_id": strict_result.paper_id,
                "chunk_id": strict_result.chunk_id,
                "document_id": strict_result.document_id,
                "document_role": strict_result.document_role,
                "section": strict_result.section,
                "raw_output_path": str(raw_output_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "attempts": attempts,
                **main_metadata,
            }
        except Exception as error:
            attempts.extend(candidate_repair_usages)
            return {
                "status": "failed",
                "failure_class": "technical_recovery_runtime",
                "paper_id": strict_result.paper_id,
                "chunk_id": strict_result.chunk_id,
                "document_id": strict_result.document_id,
                "document_role": strict_result.document_role,
                "section": strict_result.section,
                "raw_output_path": str(raw_output_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "attempts": attempts,
                **main_metadata,
            }

        attempts.extend(candidate_repair_usages)

        raw_result = recovered.graph
        raw_output_path.write_text(
            json.dumps(
                raw_result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        recovery_payload = {
            "bridge_recovery_version": (
                BRIDGE_RECOVERY_VERSION
            ),
            "bridge_recovery_prompt_version": (
                bridge_adapter.recovery_prompt_version
            ),
            "source_reconciliation_version": (
                BRIDGE_SOURCE_RECONCILIATION_VERSION
            ),
            "paper_id": raw_result.paper_id,
            "chunk_id": raw_result.chunk_id,
            "generated_concept_count": (
                recovered.generated_concept_count
            ),
            "generated_link_count": (
                recovered.generated_link_count
            ),
            "accepted_concept_count": len(
                raw_result.concepts
            ),
            "accepted_link_count": len(
                raw_result.links
            ),
            "normalization_count": len(
                recovered.normalization_operations
            ),
            "normalization_operations": (
                recovered.normalization_operations
            ),
            "candidate_repair_attempts": (
                recovered.candidate_repair_attempts
            ),
            "repaired_candidate_count": (
                recovered.repaired_candidate_count
            ),
            "candidate_repairs": (
                recovered.candidate_repairs
            ),
            "quarantined_candidate_count": (
                recovered.quarantined_candidate_count
            ),
            "quarantined_link_count": (
                recovered.quarantined_link_count
            ),
            "quarantine_path": (
                str(quarantine_path)
                if recovered.quarantined_items
                else ""
            ),
        }
        recovery_path.write_text(
            json.dumps(
                recovery_payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if recovered.quarantined_items:
            quarantine_path.write_text(
                json.dumps(
                    {
                        "paper_id": raw_result.paper_id,
                        "chunk_id": raw_result.chunk_id,
                        "items": recovered.quarantined_items,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif quarantine_path.exists():
            quarantine_path.unlink()

        return {
            "status": "success",
            "paper_id": raw_result.paper_id,
            "chunk_id": raw_result.chunk_id,
            "document_id": raw_result.document_id,
            "document_role": raw_result.document_role,
            "section": raw_result.section,
            "raw_output_path": str(raw_output_path),
            "raw_concept_count": len(
                raw_result.concepts
            ),
            "raw_pattern_count": sum(
                concept.retention_lane
                == "accepted_pattern"
                for concept in raw_result.concepts
            ),
            "raw_frontier_count": sum(
                concept.retention_lane
                == "paper_local_frontier"
                for concept in raw_result.concepts
            ),
            "raw_link_count": len(
                raw_result.links
            ),
            "generated_concept_count": (
                recovered.generated_concept_count
            ),
            "generated_link_count": (
                recovered.generated_link_count
            ),
            "normalization_count": len(
                recovered.normalization_operations
            ),
            "candidate_repair_attempts": (
                recovered.candidate_repair_attempts
            ),
            "repaired_candidate_count": (
                recovered.repaired_candidate_count
            ),
            "quarantined_candidate_count": (
                recovered.quarantined_candidate_count
            ),
            "quarantined_link_count": (
                recovered.quarantined_link_count
            ),
            "recovery_path": str(recovery_path),
            "quarantine_path": (
                str(quarantine_path)
                if recovered.quarantined_items
                else ""
            ),
            "validation_repairs": attempt,
            "attempts": attempts,
            **main_metadata,
        }

    raise RuntimeError(
        "Unexpected raw Bridge extraction loop exit."
    )
