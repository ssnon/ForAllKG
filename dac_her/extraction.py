from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dac_her.chunking import ChunkSpec, count_tokens
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.llm_openrouter import OpenRouterLLM
from dac_her.prompts import SYSTEM_PROMPT, build_extraction_prompt
from dac_her.schemas import KnowledgeGraph
from dac_her.validation import validate_graph_provenance
from dac_her.graph_normalization import normalize_graph_vocabularies
from dac_her.measurement_scalarization import (
    format_scalarization_errors,
    measurement_scalarization_issues,
)
from dac_her.vocab_registry import VocabularyRegistry


def chunk_output_path(
    chunk: ChunkSpec,
    chunk_output_dir: str | Path,
) -> Path:
    safe_chunk_id = chunk.chunk_id.replace(":", "__")
    return Path(chunk_output_dir) / f"{safe_chunk_id}.json"


def is_truncation_error(
    error: Exception,
    usage: dict[str, Any],
) -> bool:
    message = str(error).lower()
    finish_reason = usage.get("finish_reason")
    output_tokens = usage.get("output_tokens")
    max_tokens = usage.get("max_completion_tokens")

    if finish_reason == "length":
        return True
    if "eof while parsing" in message or "truncated" in message:
        return True

    return (
        isinstance(output_tokens, int)
        and isinstance(max_tokens, int)
        and output_tokens >= max_tokens
    )


def load_existing_result(
    *,
    chunk: ChunkSpec,
    output_path: str | Path,
) -> KnowledgeGraph | None:
    output_path = Path(output_path)
    if not output_path.exists():
        return None

    try:
        result = KnowledgeGraph.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        validate_graph_provenance(
            result,
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
            section=chunk.section,
            document_id=chunk.document_id,
            document_role=chunk.document_role,
            page_ids=chunk.page_ids,
            asset_ids=chunk.asset_ids,
        )
        return result
    except Exception:
        return None


def extract_one_chunk(
    *,
    chunk: ChunkSpec,
    model: str,
    provider: str | None,
    policy: ExtractionPolicy,
    chunk_output_dir: str | Path,
    debug_dir: str | Path,
    experiment_registry: VocabularyRegistry,
    metric_registry: VocabularyRegistry,
    vocabulary_context: str,
    force: bool = False,
) -> dict[str, Any]:
    output_path = chunk_output_path(chunk, chunk_output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = load_existing_result(
            chunk=chunk,
            output_path=output_path,
        )
        if existing is not None:
            return {
                "status": "skipped",
                "paper_id": chunk.paper_id,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "document_id": chunk.document_id,
                "document_role": chunk.document_role,
                "page_ids": list(chunk.page_ids),
                "asset_ids": list(chunk.asset_ids),
                "chunk_index": chunk.index,
                "split_depth": chunk.split_depth,
                "source_characters": len(chunk.core_text),
                "source_tokens_estimated": count_tokens(chunk.core_text),
                "output_path": str(output_path),
                "node_count": len(existing.all_node_ids()),
                "edge_count": len(existing.edges),
            }

    validation_feedback: str | None = None
    attempt_usages: list[dict[str, Any]] = []

    for attempt in range(policy.max_semantic_repairs + 1):
        llm = OpenRouterLLM(
            model=model,
            provider=provider,
            reproducible=False,
            zdr=True,
        )

        safe_chunk_id = chunk.chunk_id.replace(":", "__")
        debug_path = (
            debug_dir
            / f"{safe_chunk_id}__attempt_{attempt}.json"
        )

        try:
            result = llm.generate_structured(
                system_prompt=SYSTEM_PROMPT,
                prompt=build_extraction_prompt(
                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_role=chunk.document_role,
                    section=chunk.section,
                    page_ids=chunk.page_ids,
                    asset_ids=chunk.asset_ids,
                    asset_context=chunk.asset_context,
                    vocabulary_context=vocabulary_context,
                    left_context=chunk.left_context,
                    core_text=chunk.core_text,
                    right_context=chunk.right_context,
                    validation_feedback=validation_feedback,
                ),
                response_model=KnowledgeGraph,
                temperature=0.0,
                max_tokens=policy.max_completion_tokens,
                debug_path=debug_path,
            )

            result, vocabulary_issues = normalize_graph_vocabularies(
                result,
                experiment_registry=experiment_registry,
                metric_registry=metric_registry,
            )

            scalar_issues = measurement_scalarization_issues(result)
            if scalar_issues:
                debug_path.write_text(
                    json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise ValueError(format_scalarization_errors(scalar_issues))

            try:
                validate_graph_provenance(
                    result,
                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    section=chunk.section,
                    document_id=chunk.document_id,
                    document_role=chunk.document_role,
                    page_ids=chunk.page_ids,
                    asset_ids=chunk.asset_ids,
                )
            except ValueError:
                debug_path.write_text(
                    json.dumps(
                        result.model_dump(),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise

            usage = dict(llm.last_call_metadata or {})
            usage["max_completion_tokens"] = policy.max_completion_tokens
            attempt_usages.append(usage)

            output_path.write_text(
                json.dumps(
                    result.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            output_tokens = usage.get("output_tokens")
            utilization = (
                output_tokens / policy.max_completion_tokens
                if isinstance(output_tokens, int)
                else None
            )

            return {
                "status": "success",
                "paper_id": chunk.paper_id,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "document_id": chunk.document_id,
                "document_role": chunk.document_role,
                "page_ids": list(chunk.page_ids),
                "asset_ids": list(chunk.asset_ids),
                "chunk_index": chunk.index,
                "split_depth": chunk.split_depth,
                "semantic_repairs": attempt,
                "api_attempts": attempt + 1,
                "source_characters": len(chunk.core_text),
                "source_tokens_estimated": count_tokens(chunk.core_text),
                "output_path": str(output_path),
                "utilization": utilization,
                "node_count": len(result.all_node_ids()),
                "edge_count": len(result.edges),
                "unregistered_vocabulary_count": len(vocabulary_issues),
                "vocabulary_issues": [item.to_dict() for item in vocabulary_issues],
                "attempt_usages": attempt_usages,
                **usage,
            }

        except Exception as error:
            usage = dict(llm.last_call_metadata or {})
            usage["max_completion_tokens"] = policy.max_completion_tokens
            attempt_usages.append(usage)

            if is_truncation_error(error, usage):
                source_tokens = count_tokens(chunk.core_text)

                # A very small source chunk can still hit the output limit when
                # the model emits a bloated or repetitive graph. Splitting such
                # a chunk is unsafe and used to crash the whole paper run. Give
                # the model one or more compact retries before asking the outer
                # runner to split the source.
                if source_tokens < 600 and attempt < policy.max_semantic_repairs:
                    validation_feedback = (
                        "The previous response hit the completion-token limit "
                        "for a small source chunk. Return a drastically smaller "
                        "complete graph. Extract only the highest-value facts "
                        "explicitly stated in CORE_TEXT. Omit background, repeated "
                        "claims, weak paraphrases, and low-value comparison nodes. "
                        "Hard compact limits for this retry: at most 12 entities, "
                        "6 experiments, 3 calculations, 12 measurements, 5 "
                        "observation claims, 4 mechanism claims, 2 measurement "
                        "groups, and 45 edges. Keep evidence_text concise and use "
                        "only directly supporting evidence pointers."
                    )
                    print(
                        "[COMPACT RETRY AFTER TRUNCATION]",
                        chunk.chunk_id,
                        f"attempt={attempt + 1}",
                        f"source_tokens={source_tokens}",
                        flush=True,
                    )
                    continue

                return {
                    "status": "truncated",
                    "paper_id": chunk.paper_id,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "document_id": chunk.document_id,
                    "document_role": chunk.document_role,
                    "page_ids": list(chunk.page_ids),
                    "asset_ids": list(chunk.asset_ids),
                    "chunk_index": chunk.index,
                    "split_depth": chunk.split_depth,
                    "source_tokens_estimated": source_tokens,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "debug_path": str(debug_path),
                    "attempt_usages": attempt_usages,
                    **usage,
                }

            if (
                isinstance(error, (ValidationError, ValueError))
                and attempt < policy.max_semantic_repairs
            ):
                validation_feedback = str(error)
                print(
                    "[SEMANTIC REPAIR]",
                    chunk.chunk_id,
                    f"attempt={attempt + 1}",
                    flush=True,
                )
                continue

            return {
                "status": "failed",
                "paper_id": chunk.paper_id,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "document_id": chunk.document_id,
                "document_role": chunk.document_role,
                "page_ids": list(chunk.page_ids),
                "asset_ids": list(chunk.asset_ids),
                "chunk_index": chunk.index,
                "split_depth": chunk.split_depth,
                "semantic_repairs": attempt,
                "api_attempts": attempt + 1,
                "source_tokens_estimated": count_tokens(chunk.core_text),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "debug_path": str(debug_path),
                "attempt_usages": attempt_usages,
                **usage,
            }

    raise RuntimeError("Unexpected extraction loop exit.")
