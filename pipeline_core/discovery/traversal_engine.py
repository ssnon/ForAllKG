from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import networkx as nx

from pipeline_core.discovery.traversal_policy import candidate_edge_count, path_allowed, path_cost

TraversalAlgorithm = Literal["shortest", "top_n", "bounded_dfs", "semantic_stop"]


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


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
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def _is_alignment(attrs: dict[str, Any]) -> bool:
    return str(attrs.get("graph_layer", "")) == "corpus_alignment" or str(attrs.get("edge_class", "")) in {"registry_alignment", "pattern_alignment"}

def _paper_from_node_id(node_id: str) -> str | None:
    if not node_id.startswith("paper::"):
        return None
    parts = node_id.split("::", 2)
    if len(parts) < 3:
        return None
    paper_id = parts[1].strip()
    return paper_id or None


def _is_alignment_node(attrs: dict[str, Any]) -> bool:
    return (
        str(attrs.get("corpus_node_kind", "")) == "alignment_hub"
        or str(attrs.get("type", "")) in {
            "CorpusAlignment",
            "CorpusPattern",
        }
        or str(attrs.get("graph_layer", "")) == "corpus_alignment"
    )



@dataclass(frozen=True)
class TraversalConstraints:
    mode: str = "mechanism"
    top_k: int = 5
    max_depth: int = 8
    semantic_stop_max_depth: int = 12
    max_alignment_edges: int = 2
    min_scientific_edges: int = 1
    max_expansions: int = 20_000

    def effective_max_depth(
        self,
        algorithm: TraversalAlgorithm,
    ) -> int:
        if algorithm == "semantic_stop":
            return int(self.semantic_stop_max_depth)
        return int(self.max_depth)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PathStep:
    source: str
    target: str
    navigation_edge_id: str
    relation: str
    edge_class: str
    exploration_cost: float
    requires_verification: bool
    traversal_direction: str
    scientific_direction: str
    selected_original_edge_id: str
    source_paper_ids: tuple[str, ...]
    alternative_count: int
    alternatives: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_paper_ids"] = list(self.source_paper_ids)
        row["alternatives"] = list(self.alternatives)
        return row


@dataclass(frozen=True)
class PathResult:
    path_id: str
    algorithm: str
    mode: str
    source: str
    target: str
    semantic_stop: str | None
    nodes: tuple[str, ...]
    steps: tuple[PathStep, ...]
    total_cost: float
    hop_count: int
    scientific_edge_count: int
    alignment_edge_count: int
    candidate_edge_count: int
    reverse_edge_count: int
    source_paper_ids: tuple[str, ...]
    cross_paper_count: int
    visited_paper_ids: tuple[str, ...]
    visited_paper_count: int
    supporting_paper_ids: tuple[str, ...]
    hub_scope_paper_ids: tuple[str, ...]
    requires_verification: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "algorithm": self.algorithm,
            "mode": self.mode,
            "source": self.source,
            "target": self.target,
            "semantic_stop": self.semantic_stop,
            "nodes": list(self.nodes),
            "steps": [step.to_dict() for step in self.steps],
            "total_cost": self.total_cost,
            "hop_count": self.hop_count,
            "scientific_edge_count": self.scientific_edge_count,
            "alignment_edge_count": self.alignment_edge_count,
            "candidate_edge_count": self.candidate_edge_count,
            "reverse_edge_count": self.reverse_edge_count,
            "source_paper_ids": list(self.source_paper_ids),
            "cross_paper_count": self.cross_paper_count,
            "visited_paper_ids": list(self.visited_paper_ids),
            "visited_paper_count": self.visited_paper_count,
            "supporting_paper_ids": list(self.supporting_paper_ids),
            "hub_scope_paper_ids": list(self.hub_scope_paper_ids),
            "requires_verification": self.requires_verification,
        }


class TraversalEngine:
    def __init__(self, graph: nx.DiGraph) -> None:
        if graph.is_multigraph() or not graph.is_directed():
            raise TypeError("TraversalEngine requires the collapsed directed NavigationGraph.")
        self.graph = graph

    def _metrics(self, path: Iterable[str]) -> dict[str, int]:
        nodes = list(path)
        alignment = reverse = 0
        for left, right in zip(nodes, nodes[1:], strict=False):
            attrs = dict(self.graph.edges[left, right])
            alignment += int(_is_alignment(attrs))
            reverse += int(_as_bool(attrs.get("reverse_navigation", False)))
        candidates = candidate_edge_count(self.graph, nodes)
        return {
            "alignment": alignment,
            "candidate": candidates,
            "scientific": max(0, len(nodes) - 1 - alignment),
            "reverse": reverse,
        }

    def _allowed(
        self,
        path: list[str],
        constraints: TraversalConstraints,
        *,
        max_depth: int | None = None,
    ) -> bool:
        effective_depth = (
            constraints.max_depth
            if max_depth is None
            else int(max_depth)
        )
        if len(path) < 2 or len(path) - 1 > effective_depth:
            return False
        if not path_allowed(self.graph, path, mode=constraints.mode):
            return False
        metrics = self._metrics(path)
        return metrics["alignment"] <= constraints.max_alignment_edges and metrics["scientific"] >= constraints.min_scientific_edges

    def _raw_shortest(self, source: str, target: str, *, limit: int) -> list[list[str]]:
        if limit <= 0 or source not in self.graph or target not in self.graph:
            return []
        try:
            iterator = nx.shortest_simple_paths(self.graph, source, target, weight="exploration_cost")
            rows: list[list[str]] = []
            for path in iterator:
                rows.append([str(x) for x in path])
                if len(rows) >= limit:
                    break
            return rows
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _shortest_or_top_n(self, source: str, target: str, *, algorithm: str, constraints: TraversalConstraints) -> list[list[str]]:
        desired = 1 if algorithm == "shortest" else constraints.top_k
        accepted: list[list[str]] = []
        for path in self._raw_shortest(source, target, limit=max(100, desired * 50)):
            if self._allowed(path, constraints):
                accepted.append(path)
                if len(accepted) >= desired:
                    break
        return accepted

    def _dfs(self, source: str, target: str, *, constraints: TraversalConstraints) -> list[list[str]]:
        if source not in self.graph or target not in self.graph:
            return []
        stack: list[tuple[str, list[str]]] = [(source, [source])]
        found: list[list[str]] = []
        expansions = 0
        while stack:
            current, path = stack.pop()
            if current == target and len(path) >= 2:
                if self._allowed(path, constraints):
                    found.append(path)
                continue
            if len(path) - 1 >= constraints.max_depth:
                continue
            for neighbor in sorted(map(str, self.graph.successors(current)), reverse=True):
                if neighbor in path:
                    continue
                expansions += 1
                if expansions > constraints.max_expansions:
                    stack.clear()
                    break
                candidate = path + [neighbor]
                if not path_allowed(self.graph, candidate, mode=constraints.mode):
                    continue
                if self._metrics(candidate)["alignment"] > constraints.max_alignment_edges:
                    continue
                stack.append((neighbor, candidate))
        found.sort(key=lambda p: (path_cost(self.graph, p), len(p), p))
        return found[: constraints.top_k]

    def _semantic_stop(self, source: str, target: str, *, stop: str, constraints: TraversalConstraints) -> list[list[str]]:
        segment_limit = max(30, constraints.top_k * 10)
        left_paths = self._raw_shortest(source, stop, limit=segment_limit)
        right_paths = self._raw_shortest(stop, target, limit=segment_limit)
        combined: dict[tuple[str, ...], list[str]] = {}
        for left in left_paths:
            for right in right_paths:
                path = left[:-1] + right
                if len(set(path)) != len(path):
                    continue
                if self._allowed(
                    path,
                    constraints,
                    max_depth=(
                        constraints.semantic_stop_max_depth
                    ),
                ):
                    combined[tuple(path)] = path
        rows = list(combined.values())
        rows.sort(key=lambda p: (path_cost(self.graph, p), len(p), p))
        return rows[: constraints.top_k]

    def _step(self, source: str, target: str) -> PathStep:
        attrs = dict(self.graph.edges[source, target])
        alternatives = tuple(x for x in _json_list(attrs.get("edge_alternatives_json", "[]")) if isinstance(x, dict))
        papers = tuple(sorted({str(x) for x in _json_list(attrs.get("source_paper_ids_json", "[]")) if str(x).strip()}))
        return PathStep(
            source=source,
            target=target,
            navigation_edge_id=str(attrs.get("edge_id", "")),
            relation=str(attrs.get("relation", "")),
            edge_class=str(attrs.get("edge_class", "")),
            exploration_cost=float(attrs.get("exploration_cost", 1.0)),
            requires_verification=_as_bool(attrs.get("requires_verification", False)),
            traversal_direction=str(attrs.get("traversal_direction", "forward")),
            scientific_direction=str(attrs.get("scientific_direction", "")),
            selected_original_edge_id=str(attrs.get("selected_original_edge_id", "")),
            source_paper_ids=papers,
            alternative_count=int(attrs.get("alternative_count", 1)),
            alternatives=alternatives,
        )

    def _materialize(self, path: list[str], *, algorithm: str, mode: str, semantic_stop: str | None) -> PathResult:
        steps = tuple(self._step(left, right) for left, right in zip(path, path[1:], strict=False))
        metrics = self._metrics(path)

        visited_papers: set[str] = set()
        supporting_papers: set[str] = set()
        hub_scope_papers: set[str] = set()

        for node_id in path:
            attrs = dict(self.graph.nodes[node_id])
            namespaced_paper = _paper_from_node_id(node_id)
            if namespaced_paper:
                visited_papers.add(namespaced_paper)

            direct = str(attrs.get("source_paper_id", "")).strip()
            if direct and not _is_alignment_node(attrs):
                visited_papers.add(direct)

            if _is_alignment_node(attrs):
                hub_scope_papers.update(
                    str(x)
                    for x in _json_list(
                        attrs.get("source_paper_ids_json", "[]")
                    )
                    if str(x).strip()
                )

        for step in steps:
            edge_attrs = dict(
                self.graph.edges[
                    step.source,
                    step.target,
                ]
            )
            if _is_alignment(edge_attrs):
                hub_scope_papers.update(step.source_paper_ids)
            else:
                supporting_papers.update(step.source_paper_ids)

        supporting_papers.update(visited_papers)

        path_id = "path:" + _stable_id(
            algorithm,
            mode,
            semantic_stop or "",
            *path,
            *[
                step.navigation_edge_id
                for step in steps
            ],
        )
        visited = tuple(sorted(visited_papers))
        supporting = tuple(sorted(supporting_papers))
        hub_scope = tuple(sorted(hub_scope_papers))

        return PathResult(
            path_id=path_id,
            algorithm=algorithm,
            mode=mode,
            source=path[0],
            target=path[-1],
            semantic_stop=semantic_stop,
            nodes=tuple(path),
            steps=steps,
            total_cost=float(path_cost(self.graph, path)),
            hop_count=len(path) - 1,
            scientific_edge_count=metrics["scientific"],
            alignment_edge_count=metrics["alignment"],
            candidate_edge_count=metrics["candidate"],
            reverse_edge_count=metrics["reverse"],
            # Backward-compatible fields now mean papers actually visited
            # by paper-local nodes, not every paper represented by a hub.
            source_paper_ids=visited,
            cross_paper_count=len(visited),
            visited_paper_ids=visited,
            visited_paper_count=len(visited),
            supporting_paper_ids=supporting,
            hub_scope_paper_ids=hub_scope,
            requires_verification=metrics["candidate"] > 0,
        )

    def traverse(
        self,
        source: str,
        target: str,
        *,
        algorithm: TraversalAlgorithm = "top_n",
        constraints: TraversalConstraints | None = None,
        semantic_stop: str | None = None,
    ) -> list[PathResult]:
        constraints = constraints or TraversalConstraints()
        if source == target:
            return []
        if algorithm == "semantic_stop":
            if not semantic_stop:
                raise ValueError("semantic_stop algorithm requires a stop node.")
            raw = self._semantic_stop(source, target, stop=semantic_stop, constraints=constraints)
        elif algorithm in {"shortest", "top_n"}:
            raw = self._shortest_or_top_n(source, target, algorithm=algorithm, constraints=constraints)
        elif algorithm == "bounded_dfs":
            raw = self._dfs(source, target, constraints=constraints)
        else:
            raise ValueError(f"Unknown traversal algorithm: {algorithm!r}")
        return [self._materialize(path, algorithm=algorithm, mode=constraints.mode, semantic_stop=semantic_stop) for path in raw]
