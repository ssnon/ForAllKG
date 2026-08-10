from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import networkx as nx

from dac_her.domain_profile import DiscoverySemantics, ScientificDomainProfile
from dac_her.domains import get_domain_profile
from dac_her.candidate_units import (
    CandidateAnchor,
    CandidateUnit,
    edge_is_alignment,
    edge_is_candidate,
    edge_is_reverse,
    node_is_candidate,
    node_label,
    paper_ids_from_node,
)


_DEFAULT_DOMAIN_PROFILE = get_domain_profile("dac_her")
_DEFAULT_DISCOVERY_SEMANTICS = _DEFAULT_DOMAIN_PROFILE.discovery
_GENERIC_ENTITY_TYPES = set(_DEFAULT_DISCOVERY_SEMANTICS.generic_entity_types)
_MECHANISM_NODE_MARKERS = _DEFAULT_DISCOVERY_SEMANTICS.mechanism_node_markers
_MECHANISM_RELATION_MARKERS = _DEFAULT_DISCOVERY_SEMANTICS.mechanism_relation_markers
_SCAFFOLD_RELATIONS = set(_DEFAULT_DISCOVERY_SEMANTICS.scaffold_relations)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _normalized_type(attrs: Mapping[str, Any]) -> str:
    return "".join(
        char for char in str(attrs.get("type", "")).upper() if char.isalnum()
    )


def _is_alignment_node(attrs: Mapping[str, Any]) -> bool:
    return (
        str(attrs.get("corpus_node_kind", "")).strip().lower() == "alignment_hub"
        or _normalized_type(attrs) in {"CORPUSALIGNMENT", "CORPUSPATTERN"}
        or str(attrs.get("graph_layer", "")).strip().lower() == "corpus_alignment"
    )


def _is_mechanism_node(node_id: str, attrs: Mapping[str, Any], *, semantics: DiscoverySemantics | None = None) -> bool:
    semantics = semantics or _DEFAULT_DISCOVERY_SEMANTICS
    node_type = _normalized_type(attrs)
    if any(marker in node_type for marker in semantics.mechanism_node_markers):
        return True
    return str(node_id).split("::")[-1].strip().lower().startswith("mech_")


def _is_generic_node(node_id: str, attrs: Mapping[str, Any], *, semantics: DiscoverySemantics | None = None) -> bool:
    semantics = semantics or _DEFAULT_DISCOVERY_SEMANTICS
    if _is_alignment_node(attrs) or _is_mechanism_node(node_id, attrs, semantics=semantics):
        return False
    return _normalized_type(attrs) in semantics.generic_entity_types


def _edge_is_mechanistic(attrs: Mapping[str, Any], *, semantics: DiscoverySemantics | None = None) -> bool:
    semantics = semantics or _DEFAULT_DISCOVERY_SEMANTICS
    relation = str(attrs.get("relation", "")).strip().upper()
    return any(marker in relation for marker in semantics.mechanism_relation_markers)


def _edge_is_scaffold(attrs: Mapping[str, Any], *, semantics: DiscoverySemantics | None = None) -> bool:
    semantics = semantics or _DEFAULT_DISCOVERY_SEMANTICS
    return str(attrs.get("relation", "")).strip().upper() in semantics.scaffold_relations


def _match_similarity(match: Mapping[str, Any]) -> float:
    raw = match.get("semantic_similarity")
    if raw is not None:
        return _clip01(float(raw))
    if bool(match.get("exact_label_match")) or bool(match.get("direct_node_id")):
        return 1.0
    return 0.0


def _endpoint_tier(source_match: Mapping[str, Any], target_match: Mapping[str, Any]) -> int:
    source_exact = bool(source_match.get("exact_label_match")) or bool(source_match.get("direct_node_id"))
    target_exact = bool(target_match.get("exact_label_match")) or bool(target_match.get("direct_node_id"))
    if source_exact and target_exact:
        return 0
    if source_exact or target_exact:
        return 1
    return 2


def _bounded_shortest_paths(
    graph: nx.DiGraph,
    start: str,
    *,
    max_hops: int,
) -> dict[str, tuple[tuple[float, tuple[str, ...]], ...]]:
    """Pareto-style min-cost paths indexed by exact hop count.

    Keeping one weighted-shortest path per node is insufficient under a hard
    *total* hop budget: a cheap 10-hop prefix can crowd out a costlier 3-hop
    prefix that is the only one leaving room for the candidate unit + suffix.
    We therefore retain the minimum-cost simple path for every reachable
    ``(node, hops)`` state and expose all hop variants to the triple selector.
    """
    if start not in graph or max_hops < 0:
        return {}
    best_state: dict[tuple[str, int], tuple[float, tuple[str, ...]]] = {
        (start, 0): (0.0, (start,))
    }
    heap: list[tuple[float, int, str, tuple[str, ...]]] = [(0.0, 0, start, (start,))]

    while heap:
        cost, hops, node, path = heapq.heappop(heap)
        current = best_state.get((node, hops))
        if current is None or cost > current[0] or (cost == current[0] and path != current[1]):
            continue
        if hops >= max_hops:
            continue
        for neighbor in graph.successors(node):
            nxt = str(neighbor)
            if nxt in path:
                continue
            attrs = dict(graph.edges[node, nxt])
            edge_cost = float(attrs.get("exploration_cost", 1.0))
            new_cost = cost + edge_cost
            new_hops = hops + 1
            new_path = path + (nxt,)
            state = (nxt, new_hops)
            previous = best_state.get(state)
            if previous is not None and (new_cost, new_path) >= (previous[0], previous[1]):
                continue
            best_state[state] = (new_cost, new_path)
            heapq.heappush(heap, (new_cost, new_hops, nxt, new_path))

    by_node: dict[str, list[tuple[float, tuple[str, ...]]]] = {}
    for (node, _hops), value in best_state.items():
        by_node.setdefault(node, []).append(value)
    return {
        node: tuple(sorted(values, key=lambda item: (len(item[1]) - 1, item[0], item[1])))
        for node, values in by_node.items()
    }


@dataclass(frozen=True)
class CandidateUnitScore:
    endpoint_relevance: float
    unit_relevance: float
    mechanistic_continuity: float
    scientific_content_density: float
    cross_paper_span: float
    generic_entity_penalty: float
    alignment_penalty: float
    reverse_penalty: float
    reaction_switch_penalty: float
    path_length_penalty: float
    total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateUnitRoute:
    route_id: str
    unit: CandidateUnit
    entry_anchor: CandidateAnchor
    exit_anchor: CandidateAnchor
    source_match: dict[str, Any]
    target_match: dict[str, Any]
    nodes: tuple[str, ...]
    total_cost: float
    hop_count: int
    score: CandidateUnitScore
    reaction_node_labels: tuple[str, ...]
    visited_paper_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "candidate_unit": {
                **self.unit.to_dict(),
                "entry_anchor_id": self.entry_anchor.node_id,
                "entry_anchor_label": self.entry_anchor.label,
                "exit_anchor_id": self.exit_anchor.node_id,
                "exit_anchor_label": self.exit_anchor.label,
                "traversal_semantics": (
                    "entry/exit are distinct grounding anchors for one unverified "
                    "candidate unit; candidate-node traversal direction is not a causal claim"
                ),
            },
            "source_match": dict(self.source_match),
            "target_match": dict(self.target_match),
            "nodes": list(self.nodes),
            "total_cost": self.total_cost,
            "hop_count": self.hop_count,
            "candidate_unit_count": 1,
            "candidate_edge_count": 2,
            "candidate_unit_selection": self.score.to_dict(),
            "reaction_node_labels": list(self.reaction_node_labels),
            "visited_paper_ids": list(self.visited_paper_ids),
        }


@dataclass(frozen=True)
class CandidateUnitSelectionPolicy:
    max_depth: int = 12
    top_k: int = 12
    max_routes_per_unit: int = 1
    max_unit_semantic_similarity: float = 0.90
    min_unit_relevance: float = 0.0
    min_selection_score: float = 0.0

    endpoint_weight: float = 0.18
    unit_relevance_weight: float = 0.24
    mechanistic_continuity_weight: float = 0.24
    scientific_density_weight: float = 0.16
    cross_paper_weight: float = 0.08

    generic_penalty_weight: float = 0.16
    alignment_penalty_weight: float = 0.10
    reverse_penalty_weight: float = 0.05
    reaction_switch_penalty_weight: float = 0.18
    path_length_penalty_weight: float = 0.08


class CandidateUnitSelector:
    """Rank ``(source, candidate-unit, target)`` triples before generation.

    The selector never treats the candidate node as evidence. It searches a
    confirmed-only prefix and suffix and inserts exactly one multi-anchor
    candidate unit between two *distinct* grounded anchors.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        confirmed_graph: nx.DiGraph,
        *,
        policy: CandidateUnitSelectionPolicy | None = None,
        unit_relevance: Mapping[str, float] | None = None,
        unit_vectors: Mapping[str, Any] | None = None,
        domain_profile: ScientificDomainProfile | None = None,
    ) -> None:
        self.graph = graph
        self.confirmed_graph = confirmed_graph
        self.policy = policy or CandidateUnitSelectionPolicy()
        self.domain_profile = domain_profile or _DEFAULT_DOMAIN_PROFILE
        self.discovery_semantics = self.domain_profile.discovery
        self.unit_relevance = {str(k): _clip01(float(v)) for k, v in dict(unit_relevance or {}).items()}
        self.unit_vectors = dict(unit_vectors or {})

    def _route_diagnostics(
        self,
        nodes: tuple[str, ...],
        *,
        candidate_id: str,
        entry_id: str,
        exit_id: str,
    ) -> tuple[float, float, float, float, float, tuple[str, ...], tuple[str, ...]]:
        steps: list[dict[str, Any]] = [
            dict(self.graph.edges[left, right])
            for left, right in zip(nodes, nodes[1:], strict=False)
        ]
        candidate_step_indices = {
            index
            for index, (left, right) in enumerate(zip(nodes, nodes[1:], strict=False))
            if (left == entry_id and right == candidate_id)
            or (left == candidate_id and right == exit_id)
        }
        confirmed_steps = [step for i, step in enumerate(steps) if i not in candidate_step_indices]
        confirmed_nodes = [node for node in nodes if node != candidate_id]

        # Mechanistic continuity around the candidate unit.
        candidate_index = nodes.index(candidate_id)
        left_nodes = nodes[:candidate_index]
        right_nodes = nodes[candidate_index + 1 :]
        left_edges = steps[: max(0, candidate_index - 1)]
        right_edges = steps[candidate_index + 1 :]
        left_mech = any(_edge_is_mechanistic(step, semantics=self.discovery_semantics) for step in left_edges) or any(
            node in self.graph and _is_mechanism_node(node, dict(self.graph.nodes[node]), semantics=self.discovery_semantics)
            for node in left_nodes
        )
        right_mech = any(_edge_is_mechanistic(step, semantics=self.discovery_semantics) for step in right_edges) or any(
            node in self.graph and _is_mechanism_node(
                node,
                dict(self.graph.nodes[node]),
                semantics=self.discovery_semantics,
            )
            for node in right_nodes
        )
        mechanistic_continuity = 1.0 if left_mech and right_mech else (0.55 if left_mech or right_mech else 0.0)

        generic_flags = []
        for node in confirmed_nodes[1:-1]:
            if node not in self.graph:
                continue
            attrs = dict(self.graph.nodes[node])
            if _is_alignment_node(attrs):
                continue
            generic_flags.append(_is_generic_node(node, attrs, semantics=self.discovery_semantics))
        generic_penalty = _clip01(sum(generic_flags) / len(generic_flags)) if generic_flags else 0.0

        alignment_count = sum(edge_is_alignment(step) for step in confirmed_steps)
        alignment_penalty = _clip01(alignment_count / len(confirmed_steps)) if confirmed_steps else 0.0
        reverse_count = sum(edge_is_reverse(step) for step in confirmed_steps)
        reverse_penalty = _clip01(reverse_count / len(confirmed_steps)) if confirmed_steps else 0.0

        scaffold_count = sum(_edge_is_scaffold(step, semantics=self.discovery_semantics) or edge_is_alignment(step) for step in confirmed_steps)
        scientific_density = _clip01(1.0 - scaffold_count / len(confirmed_steps)) if confirmed_steps else 0.0

        reaction_labels: list[str] = []
        context_types = self.discovery_semantics.normalized_context_node_types()
        for node in confirmed_nodes:
            if node not in self.graph:
                continue
            attrs = dict(self.graph.nodes[node])
            if _normalized_type(attrs) in context_types:
                reaction_labels.append(node_label(self.graph, node))
        # v1 artifact field names remain reaction_* for backward compatibility;
        # semantically these are profile-configured scientific-context switches.
        unique_reactions = tuple(sorted(set(reaction_labels)))
        reaction_switch_penalty = _clip01(max(0, len(unique_reactions) - 1) / 2.0)

        papers: set[str] = set()
        for node in nodes:
            papers.update(paper_ids_from_node(self.graph, node))
        cross_paper_span = _clip01((len(papers) - 1) / 2.0) if papers else 0.0
        return (
            mechanistic_continuity,
            scientific_density,
            generic_penalty,
            alignment_penalty,
            reverse_penalty,
            unique_reactions,
            tuple(sorted(papers)),
        )

    def _score(
        self,
        *,
        source_match: Mapping[str, Any],
        target_match: Mapping[str, Any],
        unit: CandidateUnit,
        nodes: tuple[str, ...],
        entry_id: str,
        exit_id: str,
    ) -> tuple[CandidateUnitScore, tuple[str, ...], tuple[str, ...]]:
        endpoint = _clip01((_match_similarity(source_match) + _match_similarity(target_match)) / 2.0)
        unit_relevance = self.unit_relevance.get(unit.candidate_node_id, 0.0)
        (
            continuity,
            scientific_density,
            generic_penalty,
            alignment_penalty,
            reverse_penalty,
            reaction_labels,
            papers,
        ) = self._route_diagnostics(
            nodes,
            candidate_id=unit.candidate_node_id,
            entry_id=entry_id,
            exit_id=exit_id,
        )
        reaction_penalty = _clip01(max(0, len(reaction_labels) - 1) / 2.0)
        length_penalty = _clip01((len(nodes) - 1) / max(1, self.policy.max_depth))
        cross_paper_span = _clip01((len(papers) - 1) / 2.0) if papers else 0.0
        p = self.policy
        total = _clip01(
            p.endpoint_weight * endpoint
            + p.unit_relevance_weight * unit_relevance
            + p.mechanistic_continuity_weight * continuity
            + p.scientific_density_weight * scientific_density
            + p.cross_paper_weight * cross_paper_span
            - p.generic_penalty_weight * generic_penalty
            - p.alignment_penalty_weight * alignment_penalty
            - p.reverse_penalty_weight * reverse_penalty
            - p.reaction_switch_penalty_weight * reaction_penalty
            - p.path_length_penalty_weight * length_penalty
        )
        return (
            CandidateUnitScore(
                endpoint_relevance=endpoint,
                unit_relevance=unit_relevance,
                mechanistic_continuity=continuity,
                scientific_content_density=scientific_density,
                cross_paper_span=cross_paper_span,
                generic_entity_penalty=generic_penalty,
                alignment_penalty=alignment_penalty,
                reverse_penalty=reverse_penalty,
                reaction_switch_penalty=reaction_penalty,
                path_length_penalty=length_penalty,
                total=total,
            ),
            reaction_labels,
            papers,
        )

    def enumerate_routes(
        self,
        units: Iterable[CandidateUnit],
        source_matches: Iterable[Mapping[str, Any]],
        target_matches: Iterable[Mapping[str, Any]],
    ) -> list[CandidateUnitRoute]:
        units = [unit for unit in units if unit.bridge_capable]
        sources = [dict(match) for match in source_matches if str(match.get("node_id", "")) in self.confirmed_graph]
        targets = [dict(match) for match in target_matches if str(match.get("node_id", "")) in self.confirmed_graph]
        if not sources or not targets or not units:
            return []

        prefix_cache = {
            str(match["node_id"]): _bounded_shortest_paths(
                self.confirmed_graph,
                str(match["node_id"]),
                max_hops=self.policy.max_depth - 2,
            )
            for match in sources
        }
        reversed_confirmed = self.confirmed_graph.reverse(copy=False)
        suffix_cache = {
            str(match["node_id"]): _bounded_shortest_paths(
                reversed_confirmed,
                str(match["node_id"]),
                max_hops=self.policy.max_depth - 2,
            )
            for match in targets
        }

        routes: list[CandidateUnitRoute] = []
        for unit in units:
            relevance = self.unit_relevance.get(unit.candidate_node_id, 0.0)
            if relevance < self.policy.min_unit_relevance:
                continue
            for entry in unit.anchors:
                for exit_anchor in unit.anchors:
                    if entry.node_id == exit_anchor.node_id:
                        continue
                    candidate_id = unit.candidate_node_id
                    if not self.graph.has_edge(entry.node_id, candidate_id):
                        continue
                    if not self.graph.has_edge(candidate_id, exit_anchor.node_id):
                        continue
                    first_candidate = dict(self.graph.edges[entry.node_id, candidate_id])
                    second_candidate = dict(self.graph.edges[candidate_id, exit_anchor.node_id])
                    if not edge_is_candidate(first_candidate) or edge_is_reverse(first_candidate):
                        continue
                    if not edge_is_candidate(second_candidate) or not edge_is_reverse(second_candidate):
                        continue

                    for source_match in sources:
                        source_id = str(source_match["node_id"])
                        prefix_variants = prefix_cache[source_id].get(entry.node_id, ())
                        if not prefix_variants:
                            continue
                        for target_match in targets:
                            target_id = str(target_match["node_id"])
                            suffix_variants = suffix_cache[target_id].get(exit_anchor.node_id, ())
                            if not suffix_variants:
                                continue

                            best_route: tuple[float, int, tuple[str, ...]] | None = None
                            for prefix_cost, prefix_nodes in prefix_variants:
                                prefix_hops = len(prefix_nodes) - 1
                                for suffix_cost, suffix_reverse_nodes in suffix_variants:
                                    suffix_hops = len(suffix_reverse_nodes) - 1
                                    hop_count = prefix_hops + 2 + suffix_hops
                                    if hop_count > self.policy.max_depth:
                                        continue
                                    suffix_nodes = tuple(reversed(suffix_reverse_nodes))
                                    nodes = prefix_nodes + (candidate_id,) + suffix_nodes
                                    if len(set(nodes)) != len(nodes):
                                        continue
                                    total_cost = (
                                        float(prefix_cost)
                                        + float(first_candidate.get("exploration_cost", 1.0))
                                        + float(second_candidate.get("exploration_cost", 1.0))
                                        + float(suffix_cost)
                                    )
                                    candidate = (total_cost, hop_count, nodes)
                                    if best_route is None or candidate < best_route:
                                        best_route = candidate

                            if best_route is None:
                                continue
                            total_cost, hop_count, nodes = best_route
                            score, reactions, papers = self._score(
                                source_match=source_match,
                                target_match=target_match,
                                unit=unit,
                                nodes=nodes,
                                entry_id=entry.node_id,
                                exit_id=exit_anchor.node_id,
                            )
                            if score.total < self.policy.min_selection_score:
                                continue
                            route_id = _stable_id(
                                "candidate_unit_route",
                                source_id,
                                unit.unit_id,
                                entry.node_id,
                                exit_anchor.node_id,
                                target_id,
                                *nodes,
                            )
                            routes.append(
                                CandidateUnitRoute(
                                    route_id=route_id,
                                    unit=unit,
                                    entry_anchor=entry,
                                    exit_anchor=exit_anchor,
                                    source_match=source_match,
                                    target_match=target_match,
                                    nodes=nodes,
                                    total_cost=total_cost,
                                    hop_count=hop_count,
                                    score=score,
                                    reaction_node_labels=reactions,
                                    visited_paper_ids=papers,
                                )
                            )
        routes.sort(
            key=lambda route: (
                -route.score.total,
                -route.score.unit_relevance,
                route.score.reaction_switch_penalty,
                route.total_cost,
                route.hop_count,
                route.route_id,
            )
        )
        return routes

    def select(self, routes: Iterable[CandidateUnitRoute]) -> list[CandidateUnitRoute]:
        selected: list[CandidateUnitRoute] = []
        unit_counts: dict[str, int] = {}
        selected_vectors: list[Any] = []

        def cosine(left: Any, right: Any) -> float:
            try:
                import numpy as np

                a = np.asarray(left, dtype=np.float32)
                b = np.asarray(right, dtype=np.float32)
                if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
                    return 0.0
                an = float(np.linalg.norm(a))
                bn = float(np.linalg.norm(b))
                if an <= 0 or bn <= 0:
                    return 0.0
                return _clip01(float(np.dot(a / an, b / bn)))
            except Exception:
                return 0.0

        for route in routes:
            if len(selected) >= self.policy.top_k:
                break
            unit_id = route.unit.unit_id
            if unit_counts.get(unit_id, 0) >= self.policy.max_routes_per_unit:
                continue
            vector = self.unit_vectors.get(route.unit.candidate_node_id)
            if vector is not None and selected_vectors:
                overlap = max(cosine(vector, other) for other in selected_vectors)
                if overlap > self.policy.max_unit_semantic_similarity:
                    continue
            selected.append(route)
            unit_counts[unit_id] = unit_counts.get(unit_id, 0) + 1
            if vector is not None:
                selected_vectors.append(vector)
        return selected


def endpoint_pair_payload(source_match: Mapping[str, Any], target_match: Mapping[str, Any]) -> dict[str, Any]:
    source_similarity = _match_similarity(source_match)
    target_similarity = _match_similarity(target_match)
    return {
        "source_node_id": str(source_match.get("node_id", "")),
        "target_node_id": str(target_match.get("node_id", "")),
        "semantic_tier": _endpoint_tier(source_match, target_match),
        "pair_score": (source_similarity + target_similarity) / 2.0,
        "source_similarity": source_similarity,
        "target_similarity": target_similarity,
        "selection_reason": "candidate_unit_triple",
    }
