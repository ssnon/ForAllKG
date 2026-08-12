from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

import networkx as nx

from dac_her.domains.catalysis_mechanism_graph import (
    BROAD_DIRECT_MECHANISM_RELATIONS,
    BROAD_MECHANISM_CORE_TYPES,
    BROAD_MECHANISM_NODE_TYPES,
)


BROAD_GRAPH_LAYER = "broad_mechanism_abstract"
BROAD_EVIDENCE_STATUS = "broad_abstract_support"
BROAD_PROJECTION_VERSION = "broad-mechanism-projection-v2-run-bound"


def _stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _edge_records(
    graph: nx.Graph,
) -> Iterable[tuple[str, str, str, dict[str, Any]]]:
    if graph.is_multigraph():
        for left, right, key, attrs in graph.edges(keys=True, data=True):
            yield str(left), str(right), str(key), dict(attrs)
        return
    for index, (left, right, attrs) in enumerate(graph.edges(data=True)):
        yield str(left), str(right), str(index), dict(attrs)


def _node_label(node_id: str, attrs: dict[str, Any]) -> str:
    return str(
        attrs.get("label")
        or attrs.get("statement")
        or attrs.get("name")
        or attrs.get("metric")
        or node_id
    )


def broad_node_text(node_id: str, attrs: dict[str, Any]) -> str:
    parts = [
        f"type: {attrs.get('type', 'Unknown')}",
        f"label: {_node_label(node_id, attrs)}",
        "source depth: abstract",
        "epistemic use: exploration/bridge; verify before target-domain premise use",
    ]
    description = str(attrs.get("description") or "").strip()
    statement = str(attrs.get("statement") or "").strip()
    if description:
        parts.append(f"description: {description}")
    if statement and statement != _node_label(node_id, attrs):
        parts.append(f"statement: {statement}")
    return "\n".join(parts)


def _evidence_pointers(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    raw = attrs.get("evidence_pointers_json", "[]")
    if isinstance(raw, list):
        return [dict(row) for row in raw if isinstance(row, dict)]
    try:
        payload = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return []
    return [dict(row) for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _source_edge_id(
    left: str,
    right: str,
    key: str,
    attrs: dict[str, Any],
) -> str:
    return str(
        attrs.get("edge_id")
        or f"canonical:{_stable_id(left, right, key, attrs.get('relation', ''))}"
    )


def build_broad_mechanism_projection(
    canonical_graph: nx.Graph,
    *,
    retained_node_types: frozenset[str] = BROAD_MECHANISM_NODE_TYPES,
) -> tuple[nx.MultiDiGraph, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build an abstract-level mechanism projection without Bridge expansion.

    Broad abstract edges are useful for exploration but are intentionally marked
    as requiring verification before they can support a target-domain premise.
    """
    graph_attrs = dict(canonical_graph.graph)
    graph_attrs.update({
        "graph_stage": "graphagents_broad_mechanism_projection",
        "projection_mode": "mechanism",
        "projection_version": BROAD_PROJECTION_VERSION,
        "graph_layer": BROAD_GRAPH_LAYER,
        "evidence_status": BROAD_EVIDENCE_STATUS,
        "source_depth": "abstract",
        "source_extraction_run_id": str(
            canonical_graph.graph.get("run_id") or ""
        ),
        "source_extraction_run_fingerprint": str(
            canonical_graph.graph.get("run_fingerprint") or ""
        ),
    })
    projection = nx.MultiDiGraph(**graph_attrs)

    retained_ids = {
        str(node_id)
        for node_id, attrs in canonical_graph.nodes(data=True)
        if str(attrs.get("type", "")) in retained_node_types
    }

    node_rows: list[dict[str, Any]] = []
    for node_id in sorted(retained_ids):
        attrs = dict(canonical_graph.nodes[node_id])
        attrs["graph_layer"] = BROAD_GRAPH_LAYER
        attrs["evidence_status"] = BROAD_EVIDENCE_STATUS
        attrs["requires_verification"] = True
        attrs["source_depth"] = "abstract"
        attrs["node_text"] = broad_node_text(node_id, attrs)
        projection.add_node(node_id, **attrs)
        node_rows.append({
            "node_id": node_id,
            "type": str(attrs.get("type", "")),
            "label": _node_label(node_id, attrs),
            "node_text": attrs["node_text"],
            "graph_layer": BROAD_GRAPH_LAYER,
            "evidence_status": BROAD_EVIDENCE_STATUS,
            "requires_verification": True,
            "source_depth": "abstract",
            "source_paper_id": str(
                attrs.get("paper_id")
                or canonical_graph.graph.get("paper_id", "")
            ),
        })

    evidence_rows: list[dict[str, Any]] = []
    duplicate_counter: Counter[tuple[str, str, str]] = Counter()
    for left, right, key, attrs in _edge_records(canonical_graph):
        if left not in retained_ids or right not in retained_ids:
            continue
        relation = str(attrs.get("relation", ""))
        source_edge_id = _source_edge_id(left, right, key, attrs)
        projection_edge_id = (
            f"projection:{_stable_id(left, right, relation, source_edge_id)}"
        )
        duplicate_counter[(left, right, relation)] += 1
        graph_key = (
            f"broad:{_stable_id(source_edge_id, duplicate_counter[(left, right, relation)])}"
        )
        pointers = _evidence_pointers(attrs)
        source_paper_id = str(
            attrs.get("paper_id")
            or canonical_graph.graph.get("paper_id", "")
        )
        edge_attrs = dict(attrs)
        edge_attrs.update({
            "edge_id": projection_edge_id,
            "relation": relation,
            "title": relation,
            "graph_layer": BROAD_GRAPH_LAYER,
            "evidence_status": BROAD_EVIDENCE_STATUS,
            "requires_verification": True,
            "source_depth": "abstract",
            "source_edge_ids_json": json.dumps(
                [source_edge_id], ensure_ascii=False
            ),
            "projection_edge_ids_json": json.dumps(
                [projection_edge_id], ensure_ascii=False
            ),
            "supporting_node_ids_json": json.dumps(
                [left, right], ensure_ascii=False
            ),
            "evidence_pointers_json": json.dumps(
                pointers, ensure_ascii=False
            ),
            "source_paper_ids_json": json.dumps(
                [source_paper_id] if source_paper_id else [],
                ensure_ascii=False,
            ),
            "derivation_rule": "direct_broad_abstract_canonical_edge",
            "support_count": 1,
            "exploration_cost": 1.15,
        })
        projection.add_edge(left, right, key=graph_key, **edge_attrs)
        evidence_rows.append({
            "projection_edge_id": projection_edge_id,
            "source": left,
            "target": right,
            "relation": relation,
            "graph_layer": BROAD_GRAPH_LAYER,
            "evidence_status": BROAD_EVIDENCE_STATUS,
            "requires_verification": True,
            "source_depth": "abstract",
            "source_edge_ids": [source_edge_id],
            "supporting_node_ids": [left, right],
            "source_paper_ids": [source_paper_id] if source_paper_id else [],
            "evidence_pointers": pointers,
            "derivation_rule": "direct_broad_abstract_canonical_edge",
            "mechanism_bearing": relation in BROAD_DIRECT_MECHANISM_RELATIONS,
        })

    return projection, node_rows, evidence_rows


def summarize_broad_projection(
    projection: nx.Graph,
) -> dict[str, Any]:
    node_type_counts = Counter(
        str(attrs.get("type", ""))
        for _, attrs in projection.nodes(data=True)
    )
    relation_counts = Counter(
        str(attrs.get("relation", ""))
        for _, _, attrs in projection.edges(data=True)
    )
    mechanism_edges = sum(
        count
        for relation, count in relation_counts.items()
        if relation in BROAD_DIRECT_MECHANISM_RELATIONS
    )
    core_nodes = sum(
        count
        for node_type, count in node_type_counts.items()
        if node_type in BROAD_MECHANISM_CORE_TYPES
    )
    return {
        "projection_version": BROAD_PROJECTION_VERSION,
        "source_depth": "abstract",
        "nodes": projection.number_of_nodes(),
        "edges": projection.number_of_edges(),
        "mechanism_core_nodes": core_nodes,
        "direct_mechanism_edges": mechanism_edges,
        "mechanism_edge_fraction": (
            mechanism_edges / projection.number_of_edges()
            if projection.number_of_edges()
            else 0.0
        ),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
    }
