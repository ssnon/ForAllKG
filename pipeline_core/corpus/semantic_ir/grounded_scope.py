from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.multi_root import (
    CrossRootDuplicatePolicy,
    MultiRootSemanticIRLoadResult,
    load_semantic_ir_from_roots,
)
from pipeline_core.corpus.semantic_ir.corpus_audit import DuplicatePaperPolicy
from pipeline_core.corpus.semantic_ir.schema import SemanticIRBundle
from pipeline_core.corpus.semantic_ir.task_scope import (
    TaskSemanticScopeResolution,
    resolve_graph_explorer_task_scope_from_bundles,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


GroundedScopeMode = Literal["premise_and_gap", "premise_only"]


class GroundedTaskScopeSelection(StrictModel):
    schema_version: Literal[
        "grounded-task-scope-selection-v1"
    ] = "grounded-task-scope-selection-v1"

    task_id: str = Field(min_length=1)
    scope_mode: GroundedScopeMode
    selected_statement_ids: list[str] = Field(default_factory=list)
    selected_premise_statement_ids: list[str] = Field(default_factory=list)
    selected_gap_statement_ids: list[str] = Field(default_factory=list)
    selected_support_node_ids: list[str] = Field(default_factory=list)
    selected_support_edge_ids: list[str] = Field(default_factory=list)
    selected_paper_ids: list[str] = Field(default_factory=list)

    source_is_hypothesis_context: Literal[True] = True
    packet_catalog_wholesale_included: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False


class GroundedSemanticTaskScopeResult(StrictModel):
    schema_version: Literal[
        "grounded-semantic-task-scope-v1"
    ] = "grounded-semantic-task-scope-v1"

    packet_path: str
    context_path: str
    selection: GroundedTaskScopeSelection
    roots: MultiRootSemanticIRLoadResult
    resolution: TaskSemanticScopeResolution

    broad_packet_node_count: int = Field(ge=0)
    grounded_selected_node_count: int = Field(ge=0)
    grounded_selected_edge_count: int = Field(ge=0)
    source_chunk_count: int = Field(ge=0)

    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "GroundedSemanticTaskScopeResult":
        if self.grounded_selected_node_count != len(
            self.selection.selected_support_node_ids
        ):
            raise ValueError(
                "grounded_selected_node_count must equal selected node IDs"
            )
        if self.grounded_selected_edge_count != len(
            self.selection.selected_support_edge_ids
        ):
            raise ValueError(
                "grounded_selected_edge_count must equal selected edge IDs"
            )
        if self.source_chunk_count != len(self.resolution.source_chunks):
            raise ValueError(
                "source_chunk_count must equal resolved source chunk count"
            )
        return self


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _packet_catalog(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = payload.get("evidence_catalog")
    if not isinstance(catalog, dict):
        return {}, {}
    nodes = catalog.get("nodes")
    edges = catalog.get("edges")
    return (
        nodes if isinstance(nodes, dict) else {},
        edges if isinstance(edges, dict) else {},
    )


def select_grounded_context_scope(
    *,
    packet_payload: dict[str, Any],
    context_payload: dict[str, Any],
    scope_mode: GroundedScopeMode = "premise_and_gap",
) -> tuple[GroundedTaskScopeSelection, dict[str, Any]]:
    """Reduce a GraphExplorerPacket to evidence used by HypothesisContext.

    The reduced packet retains only scientific node/edge support attached to
    eligible positive premises and, optionally, eligible research gaps. It does
    not import the packet evidence catalog wholesale.
    """

    if packet_payload.get("schema_version") != "graph-explorer-input-v1":
        raise ValueError("expected GraphExplorerPacket v1")
    if context_payload.get("schema_version") != "hypothesis-context-v1":
        raise ValueError("expected HypothesisContext v1")

    packet_id = str(packet_payload.get("packet_id", "")).strip()
    packet_sha = str(packet_payload.get("packet_sha256", "")).strip()
    context_packet_id = str(context_payload.get("source_packet_id", "")).strip()
    context_packet_sha = str(
        context_payload.get("source_packet_sha256", "")
    ).strip()
    if packet_id and context_packet_id and packet_id != context_packet_id:
        raise ValueError("HypothesisContext source_packet_id mismatch")
    if packet_sha and context_packet_sha and packet_sha != context_packet_sha:
        raise ValueError("HypothesisContext source_packet_sha256 mismatch")

    task = packet_payload.get("task") if isinstance(packet_payload.get("task"), dict) else {}
    task_id = str(context_payload.get("task_id") or task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("task_id is required")

    selected_statements: list[str] = []
    premise_statements: list[str] = []
    gap_statements: list[str] = []
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    paper_ids: set[str] = set()

    for row in context_payload.get("evidence_statements") or []:
        if not isinstance(row, dict):
            continue
        eligible_premise = bool(row.get("eligible_as_premise", False))
        eligible_gap = bool(row.get("eligible_as_gap", False))
        include = eligible_premise or (
            scope_mode == "premise_and_gap" and eligible_gap
        )
        if not include:
            continue
        statement_id = str(row.get("statement_id", "")).strip()
        if statement_id:
            selected_statements.append(statement_id)
            if eligible_premise:
                premise_statements.append(statement_id)
            if eligible_gap:
                gap_statements.append(statement_id)
        node_ids.update(
            str(value).strip()
            for value in row.get("scientific_support_node_ids") or []
            if str(value).strip()
        )
        edge_ids.update(
            str(value).strip()
            for value in row.get("scientific_support_edge_ids") or []
            if str(value).strip()
        )
        paper_ids.update(
            str(value).strip()
            for value in row.get("paper_ids") or []
            if str(value).strip()
        )

    packet_nodes, packet_edges = _packet_catalog(packet_payload)

    # Edge support itself may carry the only exact endpoint lineage. Include
    # those endpoint/supporting nodes even if a context row did not redundantly
    # list them under scientific_support_node_ids.
    for edge_id in list(edge_ids):
        edge = packet_edges.get(edge_id)
        if not isinstance(edge, dict):
            continue
        for field in ("scientific_source", "scientific_target"):
            value = str(edge.get(field, "")).strip()
            if value:
                node_ids.add(value)
        node_ids.update(
            str(value).strip()
            for value in edge.get("supporting_node_ids") or []
            if str(value).strip()
        )
        paper_ids.update(
            str(value).strip()
            for value in edge.get("source_paper_ids") or []
            if str(value).strip()
        )

    reduced_nodes = {
        node_id: packet_nodes[node_id]
        for node_id in sorted(node_ids)
        if node_id in packet_nodes
    }
    reduced_edges = {
        edge_id: packet_edges[edge_id]
        for edge_id in sorted(edge_ids)
        if edge_id in packet_edges
    }

    # task_scope's exact resolver only needs these packet surfaces. Paths and
    # direct hits are intentionally empty so unrelated packet evidence cannot
    # expand the source-chunk set again.
    reduced_payload = {
        "schema_version": "graph-explorer-input-v1",
        "packet_id": packet_payload.get("packet_id"),
        "packet_sha256": packet_payload.get("packet_sha256"),
        "task": task,
        "direct_concept_hits": [],
        "paths": [],
        "evidence_catalog": {
            "nodes": reduced_nodes,
            "edges": reduced_edges,
        },
        "alignment_contexts": [],
    }

    selection = GroundedTaskScopeSelection(
        task_id=task_id,
        scope_mode=scope_mode,
        selected_statement_ids=sorted(set(selected_statements)),
        selected_premise_statement_ids=sorted(set(premise_statements)),
        selected_gap_statement_ids=sorted(set(gap_statements)),
        selected_support_node_ids=sorted(reduced_nodes),
        selected_support_edge_ids=sorted(reduced_edges),
        selected_paper_ids=sorted(paper_ids),
    )
    return selection, reduced_payload


def resolve_grounded_semantic_task_scope(
    *,
    packet_path: str | Path,
    context_path: str | Path,
    corpus_roots: list[str | Path],
    scope_mode: GroundedScopeMode = "premise_and_gap",
    duplicate_paper_policy: DuplicatePaperPolicy = "error",
    cross_root_duplicate_policy: CrossRootDuplicatePolicy = "error",
) -> tuple[GroundedSemanticTaskScopeResult, dict[str, SemanticIRBundle]]:
    packet_payload = _read_json_object(packet_path)
    context_payload = _read_json_object(context_path)
    selection, reduced_packet = select_grounded_context_scope(
        packet_payload=packet_payload,
        context_payload=context_payload,
        scope_mode=scope_mode,
    )

    requested_papers = {
        node_id.split("::", 2)[1]
        for node_id in selection.selected_support_node_ids
        if node_id.startswith("paper::") and len(node_id.split("::", 2)) == 3
    }
    roots_result, bundles = load_semantic_ir_from_roots(
        corpus_roots,
        requested_paper_ids=requested_papers,
        duplicate_paper_policy=duplicate_paper_policy,
        cross_root_duplicate_policy=cross_root_duplicate_policy,
    )

    resolution = resolve_graph_explorer_task_scope_from_bundles(
        packet_payload=reduced_packet,
        bundles=bundles,
        packet_path=str(Path(packet_path).resolve()),
        corpus_root=";".join(roots_result.roots),
    )

    packet_nodes, _ = _packet_catalog(packet_payload)
    result = GroundedSemanticTaskScopeResult(
        packet_path=str(Path(packet_path).resolve()),
        context_path=str(Path(context_path).resolve()),
        selection=selection,
        roots=roots_result,
        resolution=resolution,
        broad_packet_node_count=len(packet_nodes),
        grounded_selected_node_count=len(selection.selected_support_node_ids),
        grounded_selected_edge_count=len(selection.selected_support_edge_ids),
        source_chunk_count=len(resolution.source_chunks),
    )
    return result, bundles
