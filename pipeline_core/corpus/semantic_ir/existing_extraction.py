from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectKind,
    SemanticObjectRef,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_NODE_COLLECTION_KINDS: tuple[tuple[str, SemanticObjectKind], ...] = (
    ("entities", "entity"),
    ("experiments", "experiment"),
    ("calculations", "calculation"),
    ("measurements", "measurement"),
    ("measurement_groups", "measurement_group"),
    ("observation_claims", "observation_claim"),
    ("mechanism_claims", "mechanism_claim"),
)


class ExistingExtractionIRImportResult(StrictModel):
    """Audit record for a no-LLM import of one existing extraction attempt."""

    schema_version: Literal[
        "existing-extraction-semantic-ir-import-v1"
    ] = "existing-extraction-semantic-ir-import-v1"

    active_chunks_path: str
    bundle: SemanticIRBundle
    active_chunk_count: int = Field(ge=0)
    source_chunk_count: int = Field(ge=0)
    chunk_record_count: int = Field(ge=0)
    node_record_count: int = Field(ge=0)
    edge_record_count: int = Field(ge=0)
    semantic_record_count: int = Field(ge=0)
    artifact_path_rebase_count: int = Field(ge=0)

    llm_calls_performed: Literal[0] = 0
    source_text_copied_into_ir: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ExistingExtractionIRImportResult":
        if self.active_chunk_count != self.source_chunk_count:
            raise ValueError(
                "every active chunk must have exactly one source chunk reference"
            )
        expected_records = (
            self.chunk_record_count
            + self.node_record_count
            + self.edge_record_count
        )
        if self.semantic_record_count != expected_records:
            raise ValueError(
                "semantic_record_count must equal chunk + node + edge records"
            )
        if self.semantic_record_count != len(self.bundle.records):
            raise ValueError(
                "semantic_record_count must equal len(bundle.records)"
            )
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_artifact_path(path: Path, *, attempt_dir: Path) -> str:
    try:
        return path.resolve().relative_to(attempt_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_recorded_artifact_path(
    raw_path: str,
    *,
    attempt_dir: Path,
    expected_subdir: str,
) -> tuple[Path, bool]:
    if not str(raw_path).strip():
        raise ValueError(f"blank artifact path for expected {expected_subdir}")

    recorded = Path(raw_path)
    candidates: list[tuple[Path, bool]] = []

    if recorded.is_absolute():
        candidates.append((recorded, False))
    else:
        candidates.extend([
            (recorded, False),
            (attempt_dir / recorded, True),
        ])

    candidates.append(
        (attempt_dir / expected_subdir / recorded.name, True)
    )

    seen: set[str] = set()
    for candidate, rebased in candidates:
        marker = str(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        if candidate.is_file():
            return candidate, rebased or candidate != recorded

    rendered = ", ".join(str(path) for path, _ in candidates)
    raise FileNotFoundError(
        f"could not resolve recorded artifact path {raw_path!r}; tried: "
        + rendered
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _validate_source_graph_identity(
    *,
    active_record: dict[str, Any],
    source_payload: dict[str, Any],
    graph: Any,
) -> None:
    expected_paper_id = str(active_record.get("paper_id", "")).strip()
    expected_chunk_id = str(active_record.get("chunk_id", "")).strip()
    if not expected_paper_id or not expected_chunk_id:
        raise ValueError("active chunk record requires paper_id and chunk_id")

    for label, payload in (
        ("source chunk", source_payload),
        ("canonical graph", graph.model_dump(mode="python")),
    ):
        if str(payload.get("paper_id", "")) != expected_paper_id:
            raise ValueError(
                f"{label} paper_id does not match active chunk record: "
                f"{expected_paper_id!r}"
            )
        if str(payload.get("chunk_id", "")) != expected_chunk_id:
            raise ValueError(
                f"{label} chunk_id does not match active chunk record: "
                f"{expected_chunk_id!r}"
            )

    graph_payload = graph.model_dump(mode="python")
    for field in (
        "document_id",
        "document_role",
        "section",
        "page_ids",
        "asset_ids",
    ):
        if field not in source_payload:
            continue
        if source_payload[field] != graph_payload.get(field):
            raise ValueError(
                f"source chunk and canonical graph disagree on {field} "
                f"for {expected_chunk_id!r}"
            )


def _chunk_record(
    *,
    graph: Any,
    source_ref: SemanticObjectRef,
) -> SemanticIRRecord:
    payload = {
        "paper_id": graph.paper_id,
        "chunk_id": graph.chunk_id,
        "section": graph.section,
        "document_id": graph.document_id,
        "document_role": graph.document_role,
        "page_ids": list(graph.page_ids),
        "asset_ids": list(graph.asset_ids),
    }
    return SemanticIRRecord(
        ref=source_ref,
        payload=payload,
        source_chunk_ref=source_ref,
    )


def _node_records(
    *,
    graph: Any,
    output_path: str,
    source_ref: SemanticObjectRef,
) -> list[SemanticIRRecord]:
    rows: list[SemanticIRRecord] = []
    for collection, object_kind in _NODE_COLLECTION_KINDS:
        for node in getattr(graph, collection):
            payload = node.model_dump(mode="json")
            rows.append(
                SemanticIRRecord(
                    ref=SemanticObjectRef(
                        paper_id=graph.paper_id,
                        chunk_id=graph.chunk_id,
                        object_kind=object_kind,
                        object_id=str(node.id),
                        source_path=output_path,
                    ),
                    payload=payload,
                    source_chunk_ref=source_ref,
                )
            )
    return rows


def _edge_records(
    *,
    graph: Any,
    output_path: str,
    source_ref: SemanticObjectRef,
) -> list[SemanticIRRecord]:
    rows: list[SemanticIRRecord] = []
    duplicate_counts: Counter[str] = Counter()

    for edge in graph.edges:
        payload = edge.model_dump(mode="json")
        payload_key = _canonical_json(payload)
        base_id = _stable_id("edge", payload_key)
        duplicate_counts[base_id] += 1
        occurrence = duplicate_counts[base_id]
        object_id = (
            base_id if occurrence == 1 else f"{base_id}:{occurrence}"
        )
        rows.append(
            SemanticIRRecord(
                ref=SemanticObjectRef(
                    paper_id=graph.paper_id,
                    chunk_id=graph.chunk_id,
                    object_kind="edge",
                    object_id=object_id,
                    source_path=output_path,
                ),
                payload=payload,
                source_chunk_ref=source_ref,
            )
        )

    return rows


def load_existing_extraction_semantic_ir(
    attempt: str | Path,
) -> ExistingExtractionIRImportResult:
    """
    Wrap a completed extraction attempt as reusable Semantic IR.

    This function performs no LLM calls and does not rewrite the canonical
    chunk graphs. Source text stays on disk and is represented only by stable
    chunk references so later enrichment can lazy-load selected chunks.
    """

    attempt_path = Path(attempt)
    active_chunks_path = (
        attempt_path / "active_chunks.json"
        if attempt_path.is_dir()
        else attempt_path
    )
    if not active_chunks_path.is_file():
        raise FileNotFoundError(active_chunks_path)

    attempt_dir = active_chunks_path.parent
    active_payload = _load_json_object(active_chunks_path)

    paper_id = str(active_payload.get("paper_id", "")).strip()
    if not paper_id:
        raise ValueError("active_chunks.json is missing paper_id")

    active_rows = active_payload.get("chunks")
    if not isinstance(active_rows, list):
        raise ValueError("active_chunks.json field 'chunks' must be a list")

    declared_count = active_payload.get("active_chunk_count")
    if declared_count is not None and int(declared_count) != len(active_rows):
        raise ValueError(
            "active_chunk_count does not match len(active_chunks.chunks)"
        )

    seen_chunk_ids: set[str] = set()
    source_refs: list[SemanticObjectRef] = []
    records: list[SemanticIRRecord] = []
    fingerprint_rows: list[tuple[str, str, str]] = []
    node_record_count = 0
    edge_record_count = 0
    rebase_count = 0

    # Import lazily here so the semantic-IR primitives remain usable without
    # importing the legacy graph schema at module import time.
    from pipeline_core.corpus.graph.strict_chunk_loading import (
        load_strict_validated_chunk_graph,
    )

    for active_record in active_rows:
        if not isinstance(active_record, dict):
            raise ValueError("active chunk entries must be JSON objects")

        row_paper_id = str(active_record.get("paper_id", "")).strip()
        chunk_id = str(active_record.get("chunk_id", "")).strip()
        if row_paper_id != paper_id:
            raise ValueError(
                "active chunk record paper_id does not match attempt paper_id"
            )
        if not chunk_id:
            raise ValueError("active chunk record is missing chunk_id")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"duplicate active chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)

        source_path, source_rebased = _resolve_recorded_artifact_path(
            str(active_record.get("source_path", "")),
            attempt_dir=attempt_dir,
            expected_subdir="source_chunks",
        )
        output_path, output_rebased = _resolve_recorded_artifact_path(
            str(active_record.get("output_path", "")),
            attempt_dir=attempt_dir,
            expected_subdir="chunks",
        )
        rebase_count += int(source_rebased) + int(output_rebased)

        source_payload = _load_json_object(source_path)
        graph = load_strict_validated_chunk_graph(output_path)
        _validate_source_graph_identity(
            active_record=active_record,
            source_payload=source_payload,
            graph=graph,
        )

        portable_source_path = _portable_artifact_path(
            source_path,
            attempt_dir=attempt_dir,
        )
        portable_output_path = _portable_artifact_path(
            output_path,
            attempt_dir=attempt_dir,
        )
        source_ref = SemanticObjectRef(
            paper_id=paper_id,
            chunk_id=chunk_id,
            object_kind="chunk",
            object_id=chunk_id,
            source_path=portable_source_path,
        )
        source_refs.append(source_ref)
        records.append(_chunk_record(graph=graph, source_ref=source_ref))

        node_rows = _node_records(
            graph=graph,
            output_path=portable_output_path,
            source_ref=source_ref,
        )
        edge_rows = _edge_records(
            graph=graph,
            output_path=portable_output_path,
            source_ref=source_ref,
        )
        records.extend(node_rows)
        records.extend(edge_rows)
        node_record_count += len(node_rows)
        edge_record_count += len(edge_rows)

        fingerprint_rows.append(
            (
                chunk_id,
                _sha256_path(source_path),
                _sha256_path(output_path),
            )
        )

    fingerprint_rows.sort()
    bundle_id = _stable_id(
        "semantic_ir_bundle",
        paper_id,
        _canonical_json(fingerprint_rows),
    )
    source_run_id = active_payload.get("run_id")

    bundle = SemanticIRBundle(
        bundle_id=bundle_id,
        paper_id=paper_id,
        source_run_id=(
            str(source_run_id) if source_run_id not in (None, "") else None
        ),
        source_attempt_directory=str(attempt_dir.resolve()),
        records=records,
        source_chunks=source_refs,
    )

    return ExistingExtractionIRImportResult(
        active_chunks_path=str(active_chunks_path.resolve()),
        bundle=bundle,
        active_chunk_count=len(active_rows),
        source_chunk_count=len(source_refs),
        chunk_record_count=len(active_rows),
        node_record_count=node_record_count,
        edge_record_count=edge_record_count,
        semantic_record_count=len(records),
        artifact_path_rebase_count=rebase_count,
    )
