from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from pipeline_core.bridge_schemas import BridgeChunkGraph
from pipeline_core.graph_io import knowledge_graph_to_networkx, save_graphml
from pipeline_core.corpus.schemas import KnowledgeGraph


def _stable_id(*parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def canonical_alias_map(graph: nx.Graph | None) -> dict[str, str]:
    if graph is None:
        return {}
    mapping: dict[str, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        canonical_id = str(node_id)
        mapping[canonical_id] = canonical_id
        raw = attrs.get("aliases_json")
        if not raw:
            continue
        try:
            aliases = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(aliases, list):
            for alias in aliases:
                mapping[str(alias)] = canonical_id
    return mapping


def build_bridge_graph(
    bridge_results: Iterable[
        BridgeChunkGraph
    ],
    *,
    strict_results: dict[
        str,
        KnowledgeGraph,
    ],
    canonical_graph: (
        nx.Graph | None
    ) = None,
    graph_layer: str = (
        "peripheral_explicit"
    ),
    evidence_status: str = (
        "source_explicit_peripheral"
    ),
) -> tuple[
    nx.MultiDiGraph,
    list[dict[str, Any]],
]:
    graph = nx.MultiDiGraph(
        graph_stage="bridge_v2",
        graph_layer=graph_layer,
        evidence_status=evidence_status,
    )
    alias_map = canonical_alias_map(canonical_graph)
    issues: list[dict[str, Any]] = []

    for result in bridge_results:
        strict_result = strict_results[result.chunk_id]
        strict_graph = knowledge_graph_to_networkx(strict_result)
        global_concept_ids = {
            concept.id: (
                f"bridge::{result.paper_id}::"
                f"{_stable_id(result.chunk_id, concept.id)}"
            )
            for concept in result.concepts
        }

        for concept in result.concepts:
            concept_id = global_concept_ids[concept.id]
            graph.add_node(
                concept_id,
                type="BridgeConcept",
                concept_type=concept.concept_type,
                label=concept.label,
                source_phrase=concept.source_phrase,
                description=concept.description or "",
                source_local_id=concept.id,
                retention_lane=concept.retention_lane,
                evidence_scope=concept.evidence_scope,
                pattern_subject=concept.pattern_subject or "",
                pattern_relation=concept.pattern_relation or "",
                pattern_object=concept.pattern_object or "",
                relation_strength=concept.relation_strength or "",
                pattern_support_mode=concept.pattern_support_mode or "",
                supporting_phrases_json=json.dumps(
                    concept.supporting_phrases, ensure_ascii=False
                ),
                subject_evidence_phrase=concept.subject_evidence_phrase or "",
                relation_evidence_phrase=concept.relation_evidence_phrase or "",
                object_evidence_phrase=concept.object_evidence_phrase or "",
                comparison_items_json=json.dumps(
                    [item.model_dump() for item in concept.comparison_items],
                    ensure_ascii=False,
                ),
                qualifiers_json=json.dumps(
                    [qualifier.model_dump() for qualifier in concept.qualifiers],
                    ensure_ascii=False,
                ),
                paper_id=result.paper_id,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_role=result.document_role,
                section=result.section,
                page_ids_json=json.dumps(result.page_ids, ensure_ascii=False),
                asset_ids_json=json.dumps(result.asset_ids, ensure_ascii=False),
                graph_layer=graph_layer,
                evidence_status=evidence_status,
            )

        for index, link in enumerate(result.links):
            raw_anchor_id = link.anchor_id
            resolved_anchor_id = alias_map.get(raw_anchor_id, raw_anchor_id)
            anchor_status = "raw_anchor"
            anchor_attrs: dict[str, Any] = {}

            if canonical_graph is not None and resolved_anchor_id in canonical_graph:
                anchor_attrs = dict(canonical_graph.nodes[resolved_anchor_id])
                anchor_status = (
                    "canonical_exact"
                    if resolved_anchor_id == raw_anchor_id
                    else "canonical_alias"
                )
            elif raw_anchor_id in strict_graph:
                anchor_attrs = dict(strict_graph.nodes[raw_anchor_id])
                if canonical_graph is not None:
                    anchor_status = "unresolved_in_canonical"
                    issues.append({
                        "issue": "bridge_anchor_missing_from_canonical",
                        "paper_id": result.paper_id,
                        "chunk_id": result.chunk_id,
                        "raw_anchor_id": raw_anchor_id,
                        "resolved_anchor_id": resolved_anchor_id,
                        "concept_id": global_concept_ids[link.concept_id],
                    })

            if resolved_anchor_id not in graph:
                anchor_node_attrs = dict(anchor_attrs)
                anchor_node_attrs["graph_layer"] = "canonical_anchor_reference"
                anchor_node_attrs["anchor_resolution_status"] = anchor_status
                graph.add_node(resolved_anchor_id, **anchor_node_attrs)

            concept_id = global_concept_ids[link.concept_id]
            edge_key = (
                f"bridge:{result.chunk_id}:{index}:"
                f"{_stable_id(raw_anchor_id, link.relation, concept_id)}"
            )
            pointer_payload = [
                pointer.model_dump() for pointer in link.evidence_pointers
            ]
            evidence_asset_ids = sorted({
                asset_id
                for pointer in link.evidence_pointers
                for asset_id in pointer.asset_ids
            })
            graph.add_edge(
                resolved_anchor_id,
                concept_id,
                key=edge_key,
                relation=link.relation,
                title=link.relation,
                paper_id=result.paper_id,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_role=result.document_role,
                section=result.section,
                subsection=link.subsection or "",
                page_ids_json=json.dumps(result.page_ids, ensure_ascii=False),
                asset_ids_json=json.dumps(result.asset_ids, ensure_ascii=False),
                evidence_pointers_json=json.dumps(
                    pointer_payload, ensure_ascii=False
                ),
                evidence_asset_ids_json=json.dumps(
                    evidence_asset_ids, ensure_ascii=False
                ),
                evidence_type=link.evidence_type,
                evidence_strength=link.evidence_strength,
                evidence_text=link.evidence_text,
                confidence=link.confidence,
                human_verified=False,
                graph_layer=graph_layer,
                evidence_status=evidence_status,
                raw_anchor_id=raw_anchor_id,
                anchor_resolution_status=anchor_status,
            )

    return graph, issues


def build_discovery_projection(
    canonical_graph: nx.Graph,
    *,
    bridge_graph: nx.Graph | None = None,
    include_bridge: bool,
) -> nx.DiGraph:
    """Backward-compatible flat projection.

    New GraphAgents integrations should prefer graphagents_adapter.py, which
    creates evidence/mechanism/exploratory projections and provenance sidecars.
    """
    source = nx.MultiDiGraph()
    source.graph.update(canonical_graph.graph)
    for node_id, attrs in canonical_graph.nodes(data=True):
        canonical_attrs = dict(attrs)
        canonical_attrs.setdefault("graph_layer", "canonical")
        canonical_attrs.setdefault("evidence_status", "source_asserted")
        source.add_node(node_id, **canonical_attrs)

    if canonical_graph.is_multigraph():
        source.add_edges_from(canonical_graph.edges(keys=True, data=True))
    else:
        for index, (left, right, attrs) in enumerate(canonical_graph.edges(data=True)):
            source.add_edge(left, right, key=str(index), **dict(attrs))

    if include_bridge:
        if bridge_graph is None:
            raise ValueError("bridge_graph is required for core-plus-bridge mode.")
        for node_id, attrs in bridge_graph.nodes(data=True):
            if node_id in source:
                merged = dict(source.nodes[node_id])
                for key, value in dict(attrs).items():
                    if key in {
                        "graph_layer",
                        "evidence_status",
                        "anchor_resolution_status",
                    }:
                        continue
                    if key not in merged or merged[key] == "" or merged[key] is None:
                        merged[key] = value
                source.nodes[node_id].update(merged)
            else:
                source.add_node(node_id, **dict(attrs))
        if bridge_graph.is_multigraph():
            iterator = bridge_graph.edges(keys=True, data=True)
        else:
            iterator = (
                (left, right, str(index), attrs)
                for index, (left, right, attrs) in enumerate(
                    bridge_graph.edges(data=True)
                )
            )
        for left, right, key, attrs in iterator:
            candidate = str(key)
            while source.has_edge(left, right, candidate):
                candidate = f"bridge:{candidate}"
            source.add_edge(left, right, key=candidate, **dict(attrs))

    projection = nx.DiGraph(
        graph_stage="discovery_projection",
        projection_mode=("core-plus-bridge" if include_bridge else "core-only"),
    )
    projection.add_nodes_from(source.nodes(data=True))

    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for left, right, key, attrs in source.edges(keys=True, data=True):
        grouped.setdefault((str(left), str(right)), []).append(
            (str(key), dict(attrs))
        )

    for (left, right), entries in grouped.items():
        relations = sorted({
            str(attrs.get("relation", "RELATED_TO"))
            for _, attrs in entries
        })
        layers = sorted({
            str(attrs.get("graph_layer", "canonical"))
            for _, attrs in entries
        })
        bridge_count = sum(
            attrs.get("graph_layer") == "peripheral_explicit"
            for _, attrs in entries
        )
        projection.add_edge(
            left,
            right,
            relation=(relations[0] if len(relations) == 1 else "MULTI_RELATION"),
            title=(relations[0] if len(relations) == 1 else " / ".join(relations)),
            relations_json=json.dumps(relations, ensure_ascii=False),
            source_edge_keys_json=json.dumps(
                [key for key, _ in entries], ensure_ascii=False
            ),
            graph_layers_json=json.dumps(layers, ensure_ascii=False),
            source_edge_count=len(entries),
            bridge_edge_count=bridge_count,
        )

    return projection


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            })


def write_bridge_tables(
    graph: nx.MultiDiGraph,
    issues: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concept_rows = [
        {"node_id": str(node_id), **dict(attrs)}
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("type") == "BridgeConcept"
    ]
    pattern_rows = [
        row for row in concept_rows
        if row.get("retention_lane") == "accepted_pattern"
    ]
    frontier_rows = [
        row for row in concept_rows
        if row.get("retention_lane") == "paper_local_frontier"
    ]
    link_rows = [
        {
            "source": str(left),
            "target": str(right),
            "key": str(key),
            **dict(attrs),
        }
        for left, right, key, attrs in graph.edges(keys=True, data=True)
    ]

    _write_rows(output_dir / "bridge_concepts.csv", concept_rows)
    _write_rows(output_dir / "bridge_patterns.csv", pattern_rows)
    _write_rows(output_dir / "bridge_frontier.csv", frontier_rows)
    _write_rows(output_dir / "bridge_links.csv", link_rows)
    _write_rows(output_dir / "bridge_issues.csv", issues)
    _write_rows(output_dir / "bridge_rejected.csv", rejections)


def save_bridge_graph(graph: nx.MultiDiGraph, path: str | Path) -> Path:
    return save_graphml(graph, path)
