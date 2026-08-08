from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx

_TRUE_VALUES = {"1", "true", "yes"}


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in _TRUE_VALUES


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return list(value)
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _stable_id(*parts: object, length: int = 20) -> str:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


@dataclass(frozen=True)
class NavigationPolicy:
    scientific_confirmed_cost: float = 1.0
    derived_mechanism_cost: float = 1.1
    registry_alignment_cost: float = 1.5
    pattern_alignment_cost: float = 1.5
    semantic_candidate_cost: float = 2.5
    reverse_penalty: float = 0.6
    synthesize_reverse_scientific_edges: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_papers(attrs: dict[str, Any]) -> list[str]:
    papers = {str(x) for x in _json_list(attrs.get("source_paper_ids_json")) if str(x).strip()}
    direct = str(attrs.get("source_paper_id", "")).strip()
    if direct:
        papers.add(direct)
    return sorted(papers)


def _edge_id(source: str, target: str, key: str, attrs: dict[str, Any]) -> str:
    return str(attrs.get("edge_id") or f"source:{_stable_id(source, target, key, attrs.get('relation', ''))}")


def _classify(graph: nx.MultiDiGraph, source: str, target: str, attrs: dict[str, Any]) -> str:
    status = str(attrs.get("evidence_status", "")).strip().lower()
    layer = str(attrs.get("graph_layer", "")).strip().lower()
    if _as_bool(attrs.get("requires_verification", False)) or status == "semantic_candidate" or layer == "bridge_candidate":
        return "semantic_candidate"

    corpus_kind = str(attrs.get("corpus_edge_kind", "")).strip().lower()
    relation = str(attrs.get("relation", "")).strip().upper()
    if corpus_kind == "alignment" or layer == "corpus_alignment":
        source_type = str(graph.nodes[source].get("type", "")) if source in graph else ""
        target_type = str(graph.nodes[target].get("type", "")) if target in graph else ""
        if "PATTERN" in relation or source_type == "CorpusPattern" or target_type == "CorpusPattern":
            return "pattern_alignment"
        return "registry_alignment"

    if status.startswith("derived") or "bridge" in status or "bridge" in layer:
        return "derived_mechanism"
    return "scientific_confirmed"


def _cost(edge_class: str, policy: NavigationPolicy) -> float:
    return {
        "scientific_confirmed": policy.scientific_confirmed_cost,
        "derived_mechanism": policy.derived_mechanism_cost,
        "registry_alignment": policy.registry_alignment_cost,
        "pattern_alignment": policy.pattern_alignment_cost,
        "semantic_candidate": policy.semantic_candidate_cost,
    }[edge_class]


def _alternative(graph: nx.MultiDiGraph, source: str, target: str, key: str, attrs: dict[str, Any], policy: NavigationPolicy, *, reverse: bool) -> dict[str, Any]:
    edge_class = _classify(graph, source, target, attrs)
    base_cost = _cost(edge_class, policy)
    nav_source, nav_target = (target, source) if reverse else (source, target)
    return {
        "navigation_source": nav_source,
        "navigation_target": nav_target,
        "original_source": source,
        "original_target": target,
        "original_edge_key": key,
        "original_edge_id": _edge_id(source, target, key, attrs),
        "relation": str(attrs.get("relation", "RELATED_TO")),
        "edge_class": edge_class,
        "traversal_direction": "reverse" if reverse else "forward",
        "reverse_navigation": reverse,
        "scientific_direction": f"{source} -> {target}",
        "base_cost": float(base_cost),
        "navigation_cost": float(base_cost + policy.reverse_penalty if reverse else base_cost),
        "requires_verification": _as_bool(attrs.get("requires_verification", False)),
        "evidence_status": str(attrs.get("evidence_status", "")),
        "graph_layer": str(attrs.get("graph_layer", "")),
        "source_paper_ids": _source_papers(attrs),
        "evidence_pointers_json": str(attrs.get("evidence_pointers_json", "[]")),
        "source_edge_ids_json": str(attrs.get("source_edge_ids_json", "[]")),
        "projection_edge_ids_json": str(attrs.get("projection_edge_ids_json", "[]")),
    }


def _edge_attrs(alternatives: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(alternatives, key=lambda x: (float(x["navigation_cost"]), bool(x["requires_verification"]), bool(x["reverse_navigation"]), str(x["original_edge_id"])))
    selected = ordered[0]
    relations = sorted({str(x["relation"]) for x in ordered if str(x["relation"]).strip()})
    original_ids = sorted({str(x["original_edge_id"]) for x in ordered})
    return {
        "edge_id": "navigation:" + _stable_id(selected["navigation_source"], selected["navigation_target"], selected["original_edge_id"]),
        "relation": str(selected["relation"]),
        "title": str(selected["relation"]),
        "relations_json": json.dumps(relations, ensure_ascii=False),
        "edge_class": str(selected["edge_class"]),
        "navigation_edge_kind": "collapsed_scientific_navigation",
        "traversal_direction": str(selected["traversal_direction"]),
        "reverse_navigation": bool(selected["reverse_navigation"]),
        "scientific_direction": str(selected["scientific_direction"]),
        "selected_original_edge_id": str(selected["original_edge_id"]),
        "selected_original_edge_key": str(selected["original_edge_key"]),
        "selected_original_source": str(selected["original_source"]),
        "selected_original_target": str(selected["original_target"]),
        "original_edge_ids_json": json.dumps(original_ids, ensure_ascii=False),
        "alternative_count": len(ordered),
        "edge_alternatives_json": json.dumps(ordered, ensure_ascii=False, sort_keys=True),
        "exploration_cost": float(selected["navigation_cost"]),
        "requires_verification": bool(selected["requires_verification"]),
        "evidence_status": str(selected["evidence_status"]),
        "graph_layer": str(selected["graph_layer"]),
        "source_paper_ids_json": json.dumps(selected["source_paper_ids"], ensure_ascii=False),
        "evidence_pointers_json": str(selected["evidence_pointers_json"]),
        "source_edge_ids_json": str(selected["source_edge_ids_json"]),
        "projection_edge_ids_json": str(selected["projection_edge_ids_json"]),
    }


def build_navigation_graph(corpus_graph: nx.Graph, *, policy: NavigationPolicy | None = None) -> tuple[nx.DiGraph, list[dict[str, Any]], dict[str, Any]]:
    """Create a traversal-safe DiGraph without destructively merging scientific evidence."""
    policy = policy or NavigationPolicy()
    source_graph = nx.MultiDiGraph(corpus_graph)

    navigation = nx.DiGraph(**{str(k): v for k, v in source_graph.graph.items()})
    navigation.graph.update({
        "graph_stage": "navigation_graph",
        "navigation_policy": json.dumps(policy.to_dict(), sort_keys=True),
        "source_graph_stage": str(source_graph.graph.get("graph_stage", "")),
        "source_nodes": source_graph.number_of_nodes(),
        "source_edges": source_graph.number_of_edges(),
    })
    for node_id, attrs in source_graph.nodes(data=True):
        copied = dict(attrs)
        copied["navigation_node"] = True
        navigation.add_node(str(node_id), **copied)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for left, right, key, attrs in source_graph.edges(keys=True, data=True):
        source, target, key_s = str(left), str(right), str(key)
        attrs_d = dict(attrs)
        forward = _alternative(source_graph, source, target, key_s, attrs_d, policy, reverse=False)
        grouped.setdefault((source, target), []).append(forward)
        if policy.synthesize_reverse_scientific_edges and source != target and forward["edge_class"] not in {"registry_alignment", "pattern_alignment"}:
            reverse = _alternative(source_graph, source, target, key_s, attrs_d, policy, reverse=True)
            grouped.setdefault((target, source), []).append(reverse)

    sidecar: list[dict[str, Any]] = []
    for (source, target), alternatives in sorted(grouped.items()):
        attrs = _edge_attrs(alternatives)
        navigation.add_edge(source, target, **attrs)
        sidecar.append({
            "navigation_edge_id": attrs["edge_id"],
            "source": source,
            "target": target,
            "selected_original_edge_id": attrs["selected_original_edge_id"],
            "edge_class": attrs["edge_class"],
            "exploration_cost": attrs["exploration_cost"],
            "requires_verification": attrs["requires_verification"],
            "alternative_count": attrs["alternative_count"],
            "alternatives": json.loads(attrs["edge_alternatives_json"]),
        })

    class_counts: dict[str, int] = {}
    reverse_count = candidate_count = 0
    for _, _, attrs in navigation.edges(data=True):
        cls = str(attrs.get("edge_class", "unknown"))
        class_counts[cls] = class_counts.get(cls, 0) + 1
        reverse_count += int(_as_bool(attrs.get("reverse_navigation", False)))
        candidate_count += int(_as_bool(attrs.get("requires_verification", False)))

    summary = {
        "source_nodes": source_graph.number_of_nodes(),
        "source_edges": source_graph.number_of_edges(),
        "navigation_nodes": navigation.number_of_nodes(),
        "navigation_edges": navigation.number_of_edges(),
        "reverse_navigation_edges": reverse_count,
        "candidate_navigation_edges": candidate_count,
        "edge_class_counts": dict(sorted(class_counts.items())),
        "policy": policy.to_dict(),
    }
    return navigation, sidecar, summary
