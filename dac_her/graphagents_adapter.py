from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Literal

import networkx as nx


ProjectionMode = Literal["evidence", "mechanism", "exploratory"]

_MECHANISM_NODE_TYPES = {
    "Catalyst",
    "CatalystModel",
    "Metal",
    "Support",
    "CoordinationMotif",
    "SynthesisMethod",
    "Precursor",
    "Reaction",
    "ReactionStep",
    "Intermediate",
    "Material",
    "ObservationClaim",
    "MechanismClaim",
    "BridgeConcept",
}

_ORIGIN_NODE_TYPES = {
    "Catalyst",
    "CatalystModel",
    "Support",
    "CoordinationMotif",
    "Material",
    "Reaction",
    "ReactionStep",
    "Intermediate",
}

_BACKTRACE_RELATIONS = {
    "HAS_MEASUREMENT",
    "EVALUATED_IN",
    "CHARACTERIZED_BY",
    "MODELED_BY",
    "APPLIES_TO",
    "SUPPORTS_CLAIM",
}


def _stable_id(*parts: object) -> str:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _edge_records(
    graph: nx.Graph,
) -> Iterable[tuple[str, str, str, dict[str, Any]]]:
    if graph.is_multigraph():
        for left, right, key, attrs in graph.edges(keys=True, data=True):
            yield str(left), str(right), str(key), dict(attrs)
    else:
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


def node_text(node_id: str, attrs: dict[str, Any]) -> str:
    node_type = str(attrs.get("type", "Unknown"))
    label = _node_label(node_id, attrs)
    parts = [f"type: {node_type}", f"label: {label}"]

    if attrs.get("description"):
        parts.append(f"description: {attrs['description']}")
    if attrs.get("statement") and str(attrs.get("statement")) != label:
        parts.append(f"statement: {attrs['statement']}")
    if attrs.get("retention_lane") == "accepted_pattern":
        parts.append(
            "pattern: "
            f"{attrs.get('pattern_subject', '')} "
            f"{attrs.get('pattern_relation', '')} "
            f"{attrs.get('pattern_object', '')}"
        )
        if attrs.get("qualifiers_json"):
            parts.append(f"qualifiers: {attrs['qualifiers_json']}")
    elif attrs.get("retention_lane") == "paper_local_frontier":
        parts.append("discovery lane: paper-local frontier concept")
    if attrs.get("evidence_scope"):
        parts.append(f"evidence scope: {attrs['evidence_scope']}")
    return "\n".join(str(part) for part in parts if str(part).strip())


def _edge_attr_id(left: str, right: str, key: str, attrs: dict[str, Any]) -> str:
    return str(
        attrs.get("edge_id")
        or f"edge:{_stable_id(left, right, key, attrs.get('relation', ''))}"
    )


def _add_projection_edge(
    projection: nx.DiGraph,
    *,
    source: str,
    target: str,
    relation: str,
    evidence_status: str,
    graph_layer: str,
    source_edge_ids: list[str],
    supporting_node_ids: list[str],
    evidence_pointers: list[Any] | None = None,
    derivation_rule: str = "",
    source_paper_ids: list[str] | None = None,
) -> str:
    edge_id = f"projection:{_stable_id(source, target, relation, evidence_status, source_edge_ids)}"
    record = {
        "edge_id": edge_id,
        "relation": relation,
        "title": relation,
        "evidence_status": evidence_status,
        "graph_layer": graph_layer,
        "source_edge_ids_json": json.dumps(source_edge_ids, ensure_ascii=False),
        "supporting_node_ids_json": json.dumps(
            supporting_node_ids, ensure_ascii=False
        ),
        "evidence_pointers_json": json.dumps(
            evidence_pointers or [], ensure_ascii=False
        ),
        "derivation_rule": derivation_rule,
        "source_paper_ids_json": json.dumps(
            sorted(set(source_paper_ids or [])), ensure_ascii=False
        ),
    }

    if projection.has_edge(source, target):
        existing = dict(projection.edges[source, target])
        relations = set(json.loads(existing.get("relations_json", "[]")))
        if not relations and existing.get("relation"):
            relations.add(str(existing["relation"]))
        relations.add(relation)

        edge_ids = set(json.loads(existing.get("projection_edge_ids_json", "[]")))
        if existing.get("edge_id"):
            edge_ids.add(str(existing["edge_id"]))
        edge_ids.add(edge_id)

        source_ids = set(json.loads(existing.get("source_edge_ids_json", "[]")))
        source_ids.update(source_edge_ids)
        statuses = set(json.loads(existing.get("evidence_statuses_json", "[]")))
        if existing.get("evidence_status"):
            statuses.add(str(existing["evidence_status"]))
        statuses.add(evidence_status)

        projection.edges[source, target].update({
            "relation": (
                next(iter(relations)) if len(relations) == 1 else "MULTI_RELATION"
            ),
            "title": " / ".join(sorted(relations)),
            "relations_json": json.dumps(sorted(relations), ensure_ascii=False),
            "projection_edge_ids_json": json.dumps(
                sorted(edge_ids), ensure_ascii=False
            ),
            "source_edge_ids_json": json.dumps(
                sorted(source_ids), ensure_ascii=False
            ),
            "evidence_statuses_json": json.dumps(
                sorted(statuses), ensure_ascii=False
            ),
        })
        return edge_id

    record["relations_json"] = json.dumps([relation], ensure_ascii=False)
    record["projection_edge_ids_json"] = json.dumps([edge_id], ensure_ascii=False)
    record["evidence_statuses_json"] = json.dumps(
        [evidence_status], ensure_ascii=False
    )
    projection.add_edge(source, target, **record)
    return edge_id


def _incoming_edges(
    graph: nx.Graph,
    node_id: str,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    if not graph.is_directed():
        return []
    if graph.is_multigraph():
        return [
            (str(left), str(right), str(key), dict(attrs))
            for left, right, key, attrs in graph.in_edges(
                node_id, keys=True, data=True
            )
        ]
    return [
        (str(left), str(right), str(index), dict(attrs))
        for index, (left, right, attrs) in enumerate(
            graph.in_edges(node_id, data=True)
        )
    ]


def _backtrace_origins(
    graph: nx.Graph,
    start_id: str,
    *,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Trace evidence/experiment/calculation nodes back to scientific entities."""
    results: list[dict[str, Any]] = []
    queue: deque[tuple[str, list[str], list[str], int]] = deque([
        (str(start_id), [str(start_id)], [], 0)
    ])
    visited: set[tuple[str, int]] = set()

    while queue:
        current, node_path, edge_path, depth = queue.popleft()
        state = (current, depth)
        if state in visited:
            continue
        visited.add(state)

        if current in graph:
            node_type = str(graph.nodes[current].get("type", ""))
            if current != str(start_id) and node_type in _ORIGIN_NODE_TYPES:
                results.append({
                    "origin_id": current,
                    "node_path": list(reversed(node_path)),
                    "edge_path": list(reversed(edge_path)),
                })
                continue

        if depth >= max_depth:
            continue

        for left, _, key, attrs in _incoming_edges(graph, current):
            relation = str(attrs.get("relation", ""))
            if relation not in _BACKTRACE_RELATIONS:
                continue
            edge_id = _edge_attr_id(left, current, key, attrs)
            queue.append((
                left,
                node_path + [left],
                edge_path + [edge_id],
                depth + 1,
            ))

    unique: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in results:
        signature = (row["origin_id"], tuple(row["edge_path"]))
        unique[signature] = row
    return list(unique.values())


def _concept_allowed(attrs: dict[str, Any], mode: ProjectionMode) -> bool:
    lane = str(attrs.get("retention_lane", ""))
    if lane == "accepted_pattern":
        return mode in {"mechanism", "exploratory"}
    if lane == "paper_local_frontier":
        return mode == "exploratory"
    return False


def build_graphagents_projection(
    canonical_graph: nx.Graph,
    *,
    bridge_graph: nx.Graph | None = None,
    mode: ProjectionMode = "mechanism",
) -> tuple[nx.DiGraph, list[dict[str, Any]], list[dict[str, Any]]]:
    if mode not in {"evidence", "mechanism", "exploratory"}:
        raise ValueError(f"Unknown projection mode: {mode!r}")

    projection = nx.DiGraph(
        graph_stage="graphagents_projection",
        projection_mode=mode,
    )
    evidence_rows: list[dict[str, Any]] = []

    if mode == "evidence":
        kept_ids = {str(node_id) for node_id in canonical_graph.nodes}
    else:
        kept_ids = {
            str(node_id)
            for node_id, attrs in canonical_graph.nodes(data=True)
            if str(attrs.get("type", "")) in _MECHANISM_NODE_TYPES
        }

    for node_id in sorted(kept_ids):
        attrs = dict(canonical_graph.nodes[node_id])
        attrs.setdefault("graph_layer", "canonical")
        attrs.setdefault("evidence_status", "source_asserted")
        attrs["node_text"] = node_text(node_id, attrs)
        projection.add_node(node_id, **attrs)

    # Preserve direct source-asserted canonical edges among retained nodes.
    for left, right, key, attrs in _edge_records(canonical_graph):
        if left not in kept_ids or right not in kept_ids:
            continue
        relation = str(attrs.get("relation", "RELATED_TO"))
        source_edge_id = _edge_attr_id(left, right, key, attrs)
        pointer_payload: list[Any] = []
        raw_pointers = attrs.get("evidence_pointers_json")
        if raw_pointers:
            try:
                parsed = json.loads(str(raw_pointers))
                if isinstance(parsed, list):
                    pointer_payload = parsed
            except json.JSONDecodeError:
                pass
        edge_id = _add_projection_edge(
            projection,
            source=left,
            target=right,
            relation=relation,
            evidence_status="source_asserted",
            graph_layer="canonical",
            source_edge_ids=[source_edge_id],
            supporting_node_ids=[left, right],
            evidence_pointers=pointer_payload,
            source_paper_ids=[str(attrs.get("paper_id", ""))],
        )
        evidence_rows.append({
            "projection_edge_id": edge_id,
            "source": left,
            "target": right,
            "relation": relation,
            "evidence_status": "source_asserted",
            "source_edge_ids": [source_edge_id],
            "supporting_node_ids": [left, right],
            "evidence_pointers": pointer_payload,
            "derivation_rule": "direct_canonical_edge",
        })

    # Mechanism/exploratory projections collapse evidence chains into safe,
    # claim-centered edges. They do not invent a causal predicate.
    if mode in {"mechanism", "exploratory"}:
        for claim_id, claim_attrs in canonical_graph.nodes(data=True):
            claim_id = str(claim_id)
            claim_type = str(claim_attrs.get("type", ""))
            if claim_type not in {"ObservationClaim", "MechanismClaim"}:
                continue
            if claim_id not in projection:
                continue

            for evidence_id, _, key, attrs in _incoming_edges(
                canonical_graph, claim_id
            ):
                if str(attrs.get("relation", "")) != "SUPPORTS_CLAIM":
                    continue
                support_edge_id = _edge_attr_id(
                    evidence_id, claim_id, key, attrs
                )
                origins = _backtrace_origins(canonical_graph, evidence_id)
                for origin in origins:
                    origin_id = str(origin["origin_id"])
                    if origin_id not in projection:
                        continue
                    relation = (
                        "SUPPORTED_OBSERVATION"
                        if claim_type == "ObservationClaim"
                        else "SUPPORTED_MECHANISM_INTERPRETATION"
                    )
                    source_edges = origin["edge_path"] + [support_edge_id]
                    node_path = origin["node_path"] + [claim_id]
                    edge_id = _add_projection_edge(
                        projection,
                        source=origin_id,
                        target=claim_id,
                        relation=relation,
                        evidence_status="derived_projection",
                        graph_layer="mechanism_projection",
                        source_edge_ids=source_edges,
                        supporting_node_ids=node_path,
                        derivation_rule="evidence_chain_to_claim",
                        source_paper_ids=[str(attrs.get("paper_id", ""))],
                    )
                    evidence_rows.append({
                        "projection_edge_id": edge_id,
                        "source": origin_id,
                        "target": claim_id,
                        "relation": relation,
                        "evidence_status": "derived_projection",
                        "source_edge_ids": source_edges,
                        "supporting_node_ids": node_path,
                        "evidence_pointers": [],
                        "derivation_rule": "evidence_chain_to_claim",
                    })

    # Add accepted bridge patterns, and frontier concepts only in explore mode.
    if bridge_graph is not None and mode in {"mechanism", "exploratory"}:
        for node_id, attrs_value in bridge_graph.nodes(data=True):
            node_id = str(node_id)
            attrs = dict(attrs_value)
            if attrs.get("type") != "BridgeConcept":
                continue
            if not _concept_allowed(attrs, mode):
                continue
            attrs["node_text"] = node_text(node_id, attrs)
            projection.add_node(node_id, **attrs)

        for anchor_id, concept_id, key, attrs in _edge_records(bridge_graph):
            if concept_id not in projection:
                continue
            relation = str(attrs.get("relation", "GROUNDS_BRIDGE_CONCEPT"))
            source_edge_id = _edge_attr_id(anchor_id, concept_id, key, attrs)
            pointer_payload: list[Any] = []
            try:
                parsed = json.loads(str(attrs.get("evidence_pointers_json", "[]")))
                if isinstance(parsed, list):
                    pointer_payload = parsed
            except json.JSONDecodeError:
                pass

            if anchor_id in projection:
                edge_id = _add_projection_edge(
                    projection,
                    source=anchor_id,
                    target=concept_id,
                    relation=relation,
                    evidence_status="source_explicit_peripheral",
                    graph_layer="bridge",
                    source_edge_ids=[source_edge_id],
                    supporting_node_ids=[anchor_id, concept_id],
                    evidence_pointers=pointer_payload,
                    source_paper_ids=[str(attrs.get("paper_id", ""))],
                )
                evidence_rows.append({
                    "projection_edge_id": edge_id,
                    "source": anchor_id,
                    "target": concept_id,
                    "relation": relation,
                    "evidence_status": "source_explicit_peripheral",
                    "source_edge_ids": [source_edge_id],
                    "supporting_node_ids": [anchor_id, concept_id],
                    "evidence_pointers": pointer_payload,
                    "derivation_rule": "direct_bridge_grounding",
                })
                continue

            # The anchor may be a Calculation/Experiment/Measurement removed
            # from the mechanism projection. Lift grounding to upstream domain
            # entities while retaining the original path in the sidecar.
            if anchor_id in canonical_graph:
                origins = _backtrace_origins(canonical_graph, anchor_id)
            else:
                origins = []
            for origin in origins:
                origin_id = str(origin["origin_id"])
                if origin_id not in projection:
                    continue
                concept_lane = str(
                    projection.nodes[concept_id].get("retention_lane", "")
                )
                lifted_relation = (
                    "GROUNDS_BRIDGE_PATTERN"
                    if concept_lane == "accepted_pattern"
                    else "GROUNDS_FRONTIER_CONCEPT"
                )
                source_edges = origin["edge_path"] + [source_edge_id]
                nodes = origin["node_path"] + [concept_id]
                edge_id = _add_projection_edge(
                    projection,
                    source=origin_id,
                    target=concept_id,
                    relation=lifted_relation,
                    evidence_status="derived_projection",
                    graph_layer="bridge_projection",
                    source_edge_ids=source_edges,
                    supporting_node_ids=nodes,
                    evidence_pointers=pointer_payload,
                    derivation_rule="lift_removed_bridge_anchor",
                    source_paper_ids=[str(attrs.get("paper_id", ""))],
                )
                evidence_rows.append({
                    "projection_edge_id": edge_id,
                    "source": origin_id,
                    "target": concept_id,
                    "relation": lifted_relation,
                    "evidence_status": "derived_projection",
                    "source_edge_ids": source_edges,
                    "supporting_node_ids": nodes,
                    "evidence_pointers": pointer_payload,
                    "derivation_rule": "lift_removed_bridge_anchor",
                })

    node_rows = [
        {
            "node_id": str(node_id),
            "type": str(attrs.get("type", "")),
            "label": _node_label(str(node_id), dict(attrs)),
            "node_text": str(attrs.get("node_text", "")),
            "graph_layer": str(attrs.get("graph_layer", "")),
            "retention_lane": str(attrs.get("retention_lane", "")),
        }
        for node_id, attrs in projection.nodes(data=True)
    ]
    return projection, node_rows, evidence_rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return path
