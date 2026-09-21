from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.corpus_audit import (
    DuplicatePaperPolicy,
    discover_extraction_attempts,
)
from pipeline_core.corpus.semantic_ir.existing_extraction import (
    load_existing_extraction_semantic_ir,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TaskSemanticResolutionMode = Literal[
    "unique_exact",
    "paper_merge_multi_mention",
    "provenance_narrowed_multi_mention",
    "collision_scoped_exact",
]


class ResolvedTaskSemanticObject(StrictModel):
    corpus_node_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    resolution_mode: TaskSemanticResolutionMode
    semantic_refs: list[SemanticObjectRef] = Field(min_length=1)
    source_chunk_refs: list[SemanticObjectRef] = Field(min_length=1)
    candidate_match_count: int = Field(ge=1)
    selected_match_count: int = Field(ge=1)
    provenance_narrowed: bool = False

    @model_validator(mode="after")
    def validate_resolution(self) -> "ResolvedTaskSemanticObject":
        if self.selected_match_count > self.candidate_match_count:
            raise ValueError(
                "selected_match_count cannot exceed candidate_match_count"
            )
        if self.selected_match_count != len(self.semantic_refs):
            raise ValueError(
                "selected_match_count must equal semantic_refs length"
            )
        if any(ref.paper_id != self.paper_id for ref in self.semantic_refs):
            raise ValueError("semantic refs must be paper-local")
        if any(ref.paper_id != self.paper_id for ref in self.source_chunk_refs):
            raise ValueError("source chunk refs must be paper-local")
        if any(ref.object_kind != "chunk" for ref in self.source_chunk_refs):
            raise ValueError("source_chunk_refs must reference chunks")
        if self.provenance_narrowed != (
            self.resolution_mode == "provenance_narrowed_multi_mention"
        ):
            raise ValueError(
                "provenance_narrowed must match the resolution mode"
            )
        return self


class TaskSemanticScopeResolution(StrictModel):
    schema_version: Literal[
        "task-semantic-scope-resolution-v2"
    ] = "task-semantic-scope-resolution-v2"

    task_id: str = Field(min_length=1)
    packet_path: str
    corpus_root: str
    referenced_corpus_node_count: int = Field(ge=0)
    paper_local_node_count: int = Field(ge=0)
    alignment_or_nonpaper_node_count: int = Field(ge=0)
    referenced_paper_ids: list[str] = Field(default_factory=list)
    available_paper_ids: list[str] = Field(default_factory=list)
    missing_paper_ids: list[str] = Field(default_factory=list)
    resolved_objects: list[ResolvedTaskSemanticObject] = Field(default_factory=list)
    unresolved_paper_local_node_ids: list[str] = Field(default_factory=list)
    ambiguous_paper_local_node_ids: list[str] = Field(default_factory=list)
    multi_mention_paper_local_node_ids: list[str] = Field(default_factory=list)
    collision_scoped_paper_local_node_ids: list[str] = Field(default_factory=list)
    provenance_narrowed_paper_local_node_ids: list[str] = Field(default_factory=list)
    source_chunks: list[SemanticObjectRef] = Field(default_factory=list)

    exact_id_resolution_only: Literal[True] = True
    paper_merge_semantics_preserved: Literal[True] = True
    text_similarity_resolution_used: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "TaskSemanticScopeResolution":
        if self.referenced_corpus_node_count != (
            self.paper_local_node_count + self.alignment_or_nonpaper_node_count
        ):
            raise ValueError(
                "referenced corpus node count must equal paper-local plus non-paper"
            )
        if set(self.missing_paper_ids) & set(self.available_paper_ids):
            raise ValueError("paper cannot be both available and missing")
        if set(self.unresolved_paper_local_node_ids) & set(
            self.ambiguous_paper_local_node_ids
        ):
            raise ValueError("node cannot be both unresolved and ambiguous")
        return self


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _paper_local_identity(node_id: str) -> tuple[str, str] | None:
    if not node_id.startswith("paper::"):
        return None
    parts = node_id.split("::", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def _collect_packet_node_ids(payload: dict[str, Any]) -> set[str]:
    node_ids: set[str] = set()

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            node_ids.add(text)

    catalog = payload.get("evidence_catalog")
    if isinstance(catalog, dict):
        nodes = catalog.get("nodes")
        if isinstance(nodes, dict):
            for node_id in nodes:
                add(node_id)
        edges = catalog.get("edges")
        if isinstance(edges, dict):
            for row in edges.values():
                if not isinstance(row, dict):
                    continue
                add(row.get("scientific_source"))
                add(row.get("scientific_target"))
                for node_id in row.get("supporting_node_ids") or []:
                    add(node_id)

    for row in payload.get("direct_concept_hits") or []:
        if isinstance(row, dict):
            add(row.get("node_id"))

    for path in payload.get("paths") or []:
        if not isinstance(path, dict):
            continue
        for node_id in path.get("node_ids") or []:
            add(node_id)
        endpoint = path.get("endpoint")
        if isinstance(endpoint, dict):
            add(endpoint.get("source_node_id"))
            add(endpoint.get("target_node_id"))
        waypoint = path.get("waypoint")
        if isinstance(waypoint, dict):
            add(waypoint.get("node_id"))
        for step in path.get("steps") or []:
            if not isinstance(step, dict):
                continue
            for field in (
                "navigation_source",
                "navigation_target",
                "scientific_source",
                "scientific_target",
            ):
                value = str(step.get(field, "")).strip()
                if value.startswith(("paper::", "corpus::")):
                    add(value)

    for context in payload.get("alignment_contexts") or []:
        if not isinstance(context, dict):
            continue
        add(context.get("hub_node_id"))
        for field in (
            "member_node_ids",
            "traversed_entry_node_ids",
            "traversed_exit_node_ids",
        ):
            for node_id in context.get(field) or []:
                add(node_id)

    return node_ids


def _packet_node_type_hints(payload: dict[str, Any]) -> dict[str, str]:
    catalog = payload.get("evidence_catalog")
    if not isinstance(catalog, dict):
        return {}
    nodes = catalog.get("nodes")
    if not isinstance(nodes, dict):
        return {}
    return {
        str(node_id): str(row.get("node_type", "")).strip()
        for node_id, row in nodes.items()
        if isinstance(row, dict) and str(row.get("node_type", "")).strip()
    }


def _packet_pointer_hints(
    payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    catalog = payload.get("evidence_catalog")
    if not isinstance(catalog, dict):
        return {}
    edges = catalog.get("edges")
    if not isinstance(edges, dict):
        return {}

    by_node: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in edges.values():
        if not isinstance(row, dict):
            continue
        pointers = [
            pointer
            for pointer in row.get("evidence_pointers") or []
            if isinstance(pointer, dict)
        ]
        if not pointers:
            continue
        node_ids = {
            str(row.get("scientific_source", "")).strip(),
            str(row.get("scientific_target", "")).strip(),
            *(
                str(node_id).strip()
                for node_id in row.get("supporting_node_ids") or []
            ),
        }
        node_ids.discard("")
        for node_id in node_ids:
            for pointer in pointers:
                signature = json.dumps(
                    pointer,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                by_node[node_id][signature] = pointer

    return {
        node_id: [rows[key] for key in sorted(rows)]
        for node_id, rows in by_node.items()
    }


def _record_graph_type(record: SemanticIRRecord) -> str:
    kind = record.ref.object_kind
    if kind == "entity":
        return str(record.payload.get("type", "")).strip()
    return {
        "experiment": "Experiment",
        "calculation": "Calculation",
        "measurement": "Measurement",
        "measurement_group": "MeasurementGroup",
        "observation_claim": "ObservationClaim",
        "mechanism_claim": "MechanismClaim",
    }.get(kind, "")


def _collision_scoped_node_id(record: SemanticIRRecord) -> str | None:
    chunk_id = str(record.ref.chunk_id or "").strip()
    local_node_id = str(record.ref.object_id).strip()
    node_type = _record_graph_type(record)
    if not chunk_id or not local_node_id or not node_type:
        return None
    digest = hashlib.sha256(
        f"{chunk_id}|{local_node_id}|{node_type}".encode("utf-8")
    ).hexdigest()[:12]
    safe_type = "".join(
        character.lower() if character.isalnum() else "_"
        for character in node_type
    ).strip("_") or "unknown"
    return f"{local_node_id}__mention_{safe_type}_{digest}"


def _bundle_node_index(
    bundle: SemanticIRBundle,
) -> dict[str, list[SemanticIRRecord]]:
    index: dict[str, list[SemanticIRRecord]] = defaultdict(list)
    for record in bundle.records:
        if record.ref.object_kind in {"chunk", "edge"}:
            continue
        index[record.ref.object_id].append(record)
        collision_id = _collision_scoped_node_id(record)
        if collision_id:
            index[collision_id].append(record)
    return dict(index)


def _chunk_payload_index(
    bundle: SemanticIRBundle,
) -> dict[tuple[str, str | None, str, str], dict[str, Any]]:
    return {
        record.ref.identity_key(): record.payload
        for record in bundle.records
        if record.ref.object_kind == "chunk"
    }


def _pointer_matches_chunk(
    pointer: dict[str, Any],
    chunk_payload: dict[str, Any],
) -> bool:
    constrained = False

    document_id = str(pointer.get("document_id", "")).strip()
    if document_id:
        constrained = True
        if document_id != str(chunk_payload.get("document_id", "")).strip():
            return False

    page_id = pointer.get("page_id")
    if page_id is not None:
        constrained = True
        if page_id not in (chunk_payload.get("page_ids") or []):
            return False

    asset_ids = {
        str(value).strip()
        for value in pointer.get("asset_ids") or []
        if str(value).strip()
    }
    if asset_ids:
        constrained = True
        chunk_assets = {
            str(value).strip()
            for value in chunk_payload.get("asset_ids") or []
            if str(value).strip()
        }
        if not (asset_ids & chunk_assets):
            return False

    return constrained


def _source_chunk_for_record(record: SemanticIRRecord) -> SemanticObjectRef:
    if record.source_chunk_ref is None:
        raise ValueError(
            "semantic node record is missing source_chunk_ref: "
            + str(record.ref.identity_key())
        )
    return record.source_chunk_ref


def _dedupe_refs(refs: list[SemanticObjectRef]) -> list[SemanticObjectRef]:
    by_key = {ref.identity_key(): ref for ref in refs}
    return [by_key[key] for key in sorted(by_key)]


def _narrow_candidates_by_packet_provenance(
    *,
    candidates: list[SemanticIRRecord],
    pointers: list[dict[str, Any]],
    chunk_payloads: dict[tuple[str, str | None, str, str], dict[str, Any]],
) -> list[SemanticIRRecord]:
    if len(candidates) <= 1 or not pointers:
        return candidates

    narrowed: list[SemanticIRRecord] = []
    for record in candidates:
        source_chunk = _source_chunk_for_record(record)
        chunk_payload = chunk_payloads.get(source_chunk.identity_key())
        if chunk_payload is None:
            continue
        if any(
            _pointer_matches_chunk(pointer, chunk_payload)
            for pointer in pointers
        ):
            narrowed.append(record)

    # Fail open to the paper-level mention set rather than guessing when packet
    # provenance cannot distinguish the merged mentions.
    return narrowed or candidates


def _resolve_payload_against_bundles(
    *,
    payload: dict[str, Any],
    bundles: dict[str, SemanticIRBundle],
    packet_path: str,
    corpus_root: str,
) -> TaskSemanticScopeResolution:
    if payload.get("schema_version") != "graph-explorer-input-v1":
        raise ValueError(
            "expected GraphExplorerPacket schema_version=graph-explorer-input-v1"
        )
    task = payload.get("task")
    if not isinstance(task, dict) or not str(task.get("task_id", "")).strip():
        raise ValueError("GraphExplorerPacket is missing task.task_id")
    task_id = str(task["task_id"])

    referenced_nodes = _collect_packet_node_ids(payload)
    paper_local = {
        node_id: identity
        for node_id in referenced_nodes
        if (identity := _paper_local_identity(node_id)) is not None
    }
    referenced_papers = sorted({paper_id for paper_id, _ in paper_local.values()})
    available_papers = sorted(set(referenced_papers) & set(bundles))
    missing_papers = sorted(set(referenced_papers) - set(bundles))

    indexes = {
        paper_id: _bundle_node_index(bundles[paper_id])
        for paper_id in available_papers
    }
    chunk_payloads = {
        paper_id: _chunk_payload_index(bundles[paper_id])
        for paper_id in available_papers
    }
    node_type_hints = _packet_node_type_hints(payload)
    pointer_hints = _packet_pointer_hints(payload)

    resolved: list[ResolvedTaskSemanticObject] = []
    unresolved: list[str] = []
    ambiguous: list[str] = []
    multi_mention: list[str] = []
    collision_scoped: list[str] = []
    provenance_narrowed: list[str] = []
    source_chunks: list[SemanticObjectRef] = []

    for corpus_node_id in sorted(paper_local):
        paper_id, source_node_id = paper_local[corpus_node_id]
        bundle = bundles.get(paper_id)
        if bundle is None:
            continue

        candidates = list(indexes[paper_id].get(source_node_id, []))
        if not candidates:
            unresolved.append(corpus_node_id)
            continue

        type_hint = node_type_hints.get(corpus_node_id, "")
        if type_hint:
            type_matches = [
                record
                for record in candidates
                if _record_graph_type(record) == type_hint
            ]
            if type_matches:
                candidates = type_matches

        kinds = {record.ref.object_kind for record in candidates}
        if len(kinds) > 1:
            ambiguous.append(corpus_node_id)
            continue

        is_collision_scoped = any(
            source_node_id == _collision_scoped_node_id(record)
            and source_node_id != record.ref.object_id
            for record in candidates
        )

        selected = _narrow_candidates_by_packet_provenance(
            candidates=candidates,
            pointers=pointer_hints.get(corpus_node_id, []),
            chunk_payloads=chunk_payloads[paper_id],
        )
        was_narrowed = len(selected) < len(candidates)

        if is_collision_scoped and len(selected) == 1:
            mode: TaskSemanticResolutionMode = "collision_scoped_exact"
            collision_scoped.append(corpus_node_id)
        elif len(candidates) == 1:
            mode = "unique_exact"
        elif was_narrowed:
            mode = "provenance_narrowed_multi_mention"
            multi_mention.append(corpus_node_id)
            provenance_narrowed.append(corpus_node_id)
        else:
            mode = "paper_merge_multi_mention"
            multi_mention.append(corpus_node_id)

        semantic_refs = _dedupe_refs([record.ref for record in selected])
        selected_source_chunks = _dedupe_refs([
            _source_chunk_for_record(record)
            for record in selected
        ])
        source_chunks.extend(selected_source_chunks)
        resolved.append(
            ResolvedTaskSemanticObject(
                corpus_node_id=corpus_node_id,
                paper_id=paper_id,
                source_node_id=source_node_id,
                resolution_mode=mode,
                semantic_refs=semantic_refs,
                source_chunk_refs=selected_source_chunks,
                candidate_match_count=len(candidates),
                selected_match_count=len(semantic_refs),
                provenance_narrowed=was_narrowed,
            )
        )

    return TaskSemanticScopeResolution(
        task_id=task_id,
        packet_path=packet_path,
        corpus_root=corpus_root,
        referenced_corpus_node_count=len(referenced_nodes),
        paper_local_node_count=len(paper_local),
        alignment_or_nonpaper_node_count=(
            len(referenced_nodes) - len(paper_local)
        ),
        referenced_paper_ids=referenced_papers,
        available_paper_ids=available_papers,
        missing_paper_ids=missing_papers,
        resolved_objects=resolved,
        unresolved_paper_local_node_ids=unresolved,
        ambiguous_paper_local_node_ids=ambiguous,
        multi_mention_paper_local_node_ids=multi_mention,
        collision_scoped_paper_local_node_ids=collision_scoped,
        provenance_narrowed_paper_local_node_ids=provenance_narrowed,
        source_chunks=_dedupe_refs(source_chunks),
    )


def resolve_graph_explorer_task_scope_from_bundles(
    *,
    packet_payload: dict[str, Any],
    bundles: dict[str, SemanticIRBundle],
    packet_path: str = "<in-memory>",
    corpus_root: str = "<in-memory>",
) -> TaskSemanticScopeResolution:
    """Resolve a packet against already-loaded paper-local Semantic IR bundles."""

    return _resolve_payload_against_bundles(
        payload=packet_payload,
        bundles=bundles,
        packet_path=packet_path,
        corpus_root=corpus_root,
    )


def resolve_graph_explorer_task_scope(
    *,
    packet_path: str | Path,
    corpus_root: str | Path,
    duplicate_paper_policy: DuplicatePaperPolicy = "error",
) -> tuple[TaskSemanticScopeResolution, dict[str, SemanticIRBundle]]:
    """
    Resolve GraphExplorerPacket evidence back to paper-local Semantic IR.

    Paper-level graph construction intentionally merges compatible same-ID
    mentions across chunks. Therefore one corpus node may legitimately resolve
    to multiple source semantic records/chunks. Those mentions remain explicit
    here rather than being mislabeled as ambiguous. Collision-scoped mention IDs
    are reconstructed deterministically from the original chunk/local ID/type.

    Resolution remains exact/provenance-based only; no text similarity is used.
    Missing extraction attempts are reported rather than treated as negative
    evidence because a discovery corpus may include papers materialized from a
    different extraction root.
    """

    packet_path = Path(packet_path)
    payload = _read_json_object(packet_path)
    referenced_nodes = _collect_packet_node_ids(payload)
    referenced_papers = sorted({
        paper_id
        for node_id in referenced_nodes
        if (identity := _paper_local_identity(node_id)) is not None
        for paper_id in [identity[0]]
    })

    discovery = discover_extraction_attempts(
        corpus_root,
        duplicate_paper_policy=duplicate_paper_policy,
    )
    attempts_by_paper = {row.paper_id: row for row in discovery.attempts}
    available_papers = sorted(set(referenced_papers) & set(attempts_by_paper))

    bundles: dict[str, SemanticIRBundle] = {}
    for paper_id in available_papers:
        imported = load_existing_extraction_semantic_ir(
            attempts_by_paper[paper_id].attempt_directory
        )
        bundles[paper_id] = imported.bundle

    result = _resolve_payload_against_bundles(
        payload=payload,
        bundles=bundles,
        packet_path=str(packet_path.resolve()),
        corpus_root=str(Path(corpus_root).resolve()),
    )
    return result, bundles
