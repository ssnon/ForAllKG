from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import networkx as nx


_TRUE_VALUES = {"1", "true", "yes"}
_ALIGNMENT_CLASSES = {"registry_alignment", "pattern_alignment"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE_VALUES


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return list(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def node_label(graph: nx.DiGraph, node_id: str) -> str:
    attrs = dict(graph.nodes[node_id]) if node_id in graph else {}
    return str(
        attrs.get("label")
        or attrs.get("statement")
        or attrs.get("name")
        or node_id
    ).strip()


def node_is_candidate(graph: nx.DiGraph, node_id: str) -> bool:
    if node_id not in graph:
        return False
    attrs = dict(graph.nodes[node_id])
    return (
        _as_bool(attrs.get("requires_verification", False))
        or str(attrs.get("policy_lane", "")).strip().lower() == "semantic_candidate"
        or str(attrs.get("evidence_status", "")).strip().lower() == "semantic_candidate"
        or str(attrs.get("graph_layer", "")).strip().lower() == "bridge_candidate"
    )


def edge_is_candidate(attrs: dict[str, Any]) -> bool:
    return (
        _as_bool(attrs.get("requires_verification", False))
        or str(attrs.get("edge_class", "")).strip().lower() == "semantic_candidate"
        or str(attrs.get("evidence_status", "")).strip().lower() == "semantic_candidate"
        or str(attrs.get("graph_layer", "")).strip().lower().startswith("bridge_candidate")
    )


def edge_is_alignment(attrs: dict[str, Any]) -> bool:
    return str(attrs.get("edge_class", "")).strip().lower() in _ALIGNMENT_CLASSES


def edge_is_reverse(attrs: dict[str, Any]) -> bool:
    return _as_bool(attrs.get("reverse_navigation", False))


def paper_ids_from_node(graph: nx.DiGraph, node_id: str) -> set[str]:
    papers: set[str] = set()
    if node_id.startswith("paper::"):
        parts = node_id.split("::", 2)
        if len(parts) >= 3 and parts[1].strip():
            papers.add(parts[1].strip())
    if node_id not in graph:
        return papers
    attrs = dict(graph.nodes[node_id])
    direct = str(attrs.get("source_paper_id", "")).strip()
    if direct:
        papers.add(direct)
    papers.update(
        str(item).strip()
        for item in _json_list(attrs.get("source_paper_ids_json", "[]"))
        if str(item).strip()
    )
    return papers


@dataclass(frozen=True)
class CandidateAnchor:
    node_id: str
    label: str
    node_type: str
    source_paper_ids: tuple[str, ...]
    forward_navigation_edge_id: str
    forward_original_edge_id: str
    forward_cost: float
    reverse_navigation_edge_id: str
    reverse_original_edge_id: str
    reverse_cost: float

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_paper_ids"] = list(self.source_paper_ids)
        return row


@dataclass(frozen=True)
class CandidateUnit:
    unit_id: str
    candidate_node_id: str
    label: str
    proposed_subject: str
    proposed_relation: str
    proposed_object: str
    source_paper_ids: tuple[str, ...]
    anchors: tuple[CandidateAnchor, ...]
    candidate_reason_codes: tuple[str, ...]

    @property
    def anchor_count(self) -> int:
        return len(self.anchors)

    @property
    def bridge_capable(self) -> bool:
        return self.anchor_count >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "candidate_node_id": self.candidate_node_id,
            "label": self.label,
            "proposed_subject": self.proposed_subject,
            "proposed_relation": self.proposed_relation,
            "proposed_object": self.proposed_object,
            "source_paper_ids": list(self.source_paper_ids),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "anchor_count": self.anchor_count,
            "bridge_capable": self.bridge_capable,
            "candidate_reason_codes": list(self.candidate_reason_codes),
            "epistemic_status": "unverified_semantic_candidate",
        }


class CandidateUnitBuilder:
    """Recover epistemic candidate units from the exploratory NavigationGraph.

    A candidate unit is a semantic-candidate BridgeConcept plus the *original*
    grounded anchors that point to it. Reverse-navigation edges are navigation
    conveniences and are never counted as independent evidence/anchors.

    The scientific candidate is the unit itself, not a single navigation edge.
    Traversing ``anchor A -> candidate -> anchor B`` therefore consumes one
    candidate unit even though it uses two candidate navigation edges.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        if graph.is_multigraph() or not graph.is_directed():
            raise TypeError("CandidateUnitBuilder requires a collapsed directed NavigationGraph.")
        self.graph = graph

    def _anchor(self, anchor_id: str, candidate_id: str) -> CandidateAnchor | None:
        if not self.graph.has_edge(anchor_id, candidate_id):
            return None
        forward = dict(self.graph.edges[anchor_id, candidate_id])
        if not edge_is_candidate(forward) or edge_is_reverse(forward):
            return None
        if not self.graph.has_edge(candidate_id, anchor_id):
            return None
        reverse = dict(self.graph.edges[candidate_id, anchor_id])
        if not edge_is_candidate(reverse) or not edge_is_reverse(reverse):
            return None
        return CandidateAnchor(
            node_id=anchor_id,
            label=node_label(self.graph, anchor_id),
            node_type=str(self.graph.nodes[anchor_id].get("type", "Unknown")),
            source_paper_ids=tuple(sorted(paper_ids_from_node(self.graph, anchor_id))),
            forward_navigation_edge_id=str(forward.get("edge_id", "")),
            forward_original_edge_id=str(forward.get("selected_original_edge_id", "")),
            forward_cost=float(forward.get("exploration_cost", 1.0)),
            reverse_navigation_edge_id=str(reverse.get("edge_id", "")),
            reverse_original_edge_id=str(reverse.get("selected_original_edge_id", "")),
            reverse_cost=float(reverse.get("exploration_cost", 1.0)),
        )

    def build(self, *, bridge_capable_only: bool = True) -> list[CandidateUnit]:
        anchors_by_candidate: dict[str, dict[str, CandidateAnchor]] = {}
        for source, target, attrs_value in self.graph.edges(data=True):
            source_id = str(source)
            target_id = str(target)
            attrs = dict(attrs_value)
            if not edge_is_candidate(attrs) or edge_is_reverse(attrs):
                continue
            if node_is_candidate(self.graph, source_id):
                continue
            if not node_is_candidate(self.graph, target_id):
                continue
            anchor = self._anchor(source_id, target_id)
            if anchor is None:
                continue
            anchors_by_candidate.setdefault(target_id, {})[source_id] = anchor

        units: list[CandidateUnit] = []
        for candidate_id, anchor_map in sorted(anchors_by_candidate.items()):
            attrs = dict(self.graph.nodes[candidate_id])
            anchors = tuple(anchor_map[node_id] for node_id in sorted(anchor_map))
            if bridge_capable_only and len(anchors) < 2:
                continue
            papers = set(paper_ids_from_node(self.graph, candidate_id))
            for anchor in anchors:
                papers.update(anchor.source_paper_ids)
            reason_codes = tuple(
                sorted(
                    str(item)
                    for item in _json_list(attrs.get("candidate_reason_codes_json", "[]"))
                    if str(item).strip()
                )
            )
            unit_id = _stable_id("candidate_unit", candidate_id, *[anchor.node_id for anchor in anchors])
            units.append(
                CandidateUnit(
                    unit_id=unit_id,
                    candidate_node_id=candidate_id,
                    label=node_label(self.graph, candidate_id),
                    proposed_subject=str(
                        attrs.get("proposed_subject")
                        or attrs.get("pattern_subject")
                        or ""
                    ),
                    proposed_relation=str(
                        attrs.get("proposed_relation")
                        or attrs.get("pattern_relation")
                        or ""
                    ),
                    proposed_object=str(
                        attrs.get("proposed_object")
                        or attrs.get("pattern_object")
                        or ""
                    ),
                    source_paper_ids=tuple(sorted(papers)),
                    anchors=anchors,
                    candidate_reason_codes=reason_codes,
                )
            )
        return units


def confirmed_navigation_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a graph in which candidate concepts/edges cannot be traversed.

    Candidate nodes are removed entirely, not merely candidate edges. This
    prevents the confirmed prefix/suffix from silently passing through one of
    the few candidate concepts that also have a merged non-candidate incident
    edge.
    """
    keep_nodes = [node for node in graph.nodes if not node_is_candidate(graph, str(node))]
    confirmed = nx.DiGraph()
    for node in keep_nodes:
        confirmed.add_node(str(node), **dict(graph.nodes[node]))
    keep_set = set(map(str, keep_nodes))
    for source, target, attrs_value in graph.edges(data=True):
        source_id = str(source)
        target_id = str(target)
        attrs = dict(attrs_value)
        if source_id not in keep_set or target_id not in keep_set:
            continue
        if edge_is_candidate(attrs):
            continue
        confirmed.add_edge(source_id, target_id, **attrs)
    return confirmed


def candidate_unit_inventory(units: Iterable[CandidateUnit]) -> dict[str, Any]:
    rows = list(units)
    anchor_histogram: dict[int, int] = {}
    for unit in rows:
        anchor_histogram[unit.anchor_count] = anchor_histogram.get(unit.anchor_count, 0) + 1
    return {
        "candidate_unit_count": len(rows),
        "bridge_capable_candidate_unit_count": sum(unit.bridge_capable for unit in rows),
        "anchor_count_histogram": {
            str(key): value for key, value in sorted(anchor_histogram.items())
        },
    }
