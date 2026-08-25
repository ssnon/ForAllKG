from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import networkx as nx

from pipeline_core.domain.domain_profile import ScientificDomainProfile
from pipeline_core.discovery.discovery_semantics import (
    is_alignment_edge as is_semantic_alignment_edge,
    is_alignment_node,
    is_generic_entity_node,
    is_mechanism_edge,
    is_mechanism_node,
    is_scaffold_edge,
    is_shared_entity_node,
    normalized_node_type,
)
from pipeline_core.discovery.candidate_units import (
    CandidateAnchor,
    CandidateUnit,
    edge_is_alignment,
    edge_is_candidate,
    edge_is_reverse,
    node_is_candidate,
    node_label,
    paper_ids_from_node,
)




def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


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
    discovery_semantics: Any | None = None,
    retain_semantic_state: bool = False,
) -> dict[str, tuple[tuple[float, tuple[str, ...]], ...]]:
    """Return bounded simple paths retained by exact routing state.

    Legacy mode retains one minimum-cost path per ``(node, hops)``.

    The corrected route contract additionally retains mechanism/alignment
    state.  This prevents a cheaper scaffold-only path from suppressing a
    scientifically richer path that reaches the same node at the same hop
    count.
    """
    if start not in graph or max_hops < 0:
        return {}

    if not retain_semantic_state:
        best_state: dict[
            tuple[str, int],
            tuple[float, tuple[str, ...]],
        ] = {
            (start, 0): (0.0, (start,))
        }

        heap: list[
            tuple[float, int, str, tuple[str, ...]]
        ] = [
            (0.0, 0, start, (start,))
        ]

        while heap:
            cost, hops, node, path = heapq.heappop(heap)

            current = best_state.get((node, hops))

            if (
                current is None
                or cost > current[0]
                or (
                    cost == current[0]
                    and path != current[1]
                )
            ):
                continue

            if hops >= max_hops:
                continue

            for neighbor in graph.successors(node):
                nxt = str(neighbor)

                if nxt in path:
                    continue

                attrs = dict(graph.edges[node, nxt])

                edge_cost = float(
                    attrs.get("exploration_cost", 1.0)
                )

                new_cost = cost + edge_cost
                new_hops = hops + 1
                new_path = path + (nxt,)

                state = (nxt, new_hops)

                previous = best_state.get(state)

                if (
                    previous is not None
                    and (new_cost, new_path)
                    >= (previous[0], previous[1])
                ):
                    continue

                best_state[state] = (
                    new_cost,
                    new_path,
                )

                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        new_hops,
                        nxt,
                        new_path,
                    ),
                )

        by_node: dict[
            str,
            list[tuple[float, tuple[str, ...]]],
        ] = {}

        for (node, _hops), value in best_state.items():
            by_node.setdefault(node, []).append(value)

        return {
            node: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        len(item[1]) - 1,
                        item[0],
                        item[1],
                    ),
                )
            )
            for node, values in by_node.items()
        }

    if discovery_semantics is None:
        raise ValueError(
            "discovery_semantics is required when "
            "retain_semantic_state=True"
        )

    start_attrs = dict(graph.nodes[start])

    start_mechanism = is_mechanism_node(
        start,
        start_attrs,
        discovery_semantics,
    )

    start_alignment = is_alignment_node(
        start_attrs
    )

    SemanticState = tuple[
        str,
        int,
        bool,
        bool,
    ]

    best_semantic_state: dict[
        SemanticState,
        tuple[float, tuple[str, ...]],
    ] = {
        (
            start,
            0,
            start_mechanism,
            start_alignment,
        ): (
            0.0,
            (start,),
        )
    }

    semantic_heap: list[
        tuple[
            float,
            int,
            str,
            bool,
            bool,
            tuple[str, ...],
        ]
    ] = [
        (
            0.0,
            0,
            start,
            start_mechanism,
            start_alignment,
            (start,),
        )
    ]

    while semantic_heap:
        (
            cost,
            hops,
            node,
            mechanism_seen,
            alignment_seen,
            path,
        ) = heapq.heappop(
            semantic_heap
        )

        state = (
            node,
            hops,
            mechanism_seen,
            alignment_seen,
        )

        current = best_semantic_state.get(
            state
        )

        if (
            current is None
            or cost > current[0]
            or (
                cost == current[0]
                and path != current[1]
            )
        ):
            continue

        if hops >= max_hops:
            continue

        for neighbor in graph.successors(node):
            nxt = str(neighbor)

            # Each side remains a simple path.  The corrected
            # carrier-reuse contract is applied only when the
            # independently simple prefix/suffix are combined.
            if nxt in path:
                continue

            attrs = dict(
                graph.edges[node, nxt]
            )

            nxt_attrs = dict(
                graph.nodes[nxt]
            )

            next_mechanism = (
                mechanism_seen
                or is_mechanism_edge(
                    attrs,
                    discovery_semantics,
                )
                or is_mechanism_node(
                    nxt,
                    nxt_attrs,
                    discovery_semantics,
                )
            )

            next_alignment = (
                alignment_seen
                or is_semantic_alignment_edge(
                    attrs
                )
                or is_alignment_node(
                    nxt_attrs
                )
            )

            new_cost = (
                cost
                + float(
                    attrs.get(
                        "exploration_cost",
                        1.0,
                    )
                )
            )

            new_hops = hops + 1
            new_path = path + (nxt,)

            next_state = (
                nxt,
                new_hops,
                next_mechanism,
                next_alignment,
            )

            previous = (
                best_semantic_state.get(
                    next_state
                )
            )

            if (
                previous is not None
                and (
                    new_cost,
                    new_path,
                )
                >= (
                    previous[0],
                    previous[1],
                )
            ):
                continue

            best_semantic_state[
                next_state
            ] = (
                new_cost,
                new_path,
            )

            heapq.heappush(
                semantic_heap,
                (
                    new_cost,
                    new_hops,
                    nxt,
                    next_mechanism,
                    next_alignment,
                    new_path,
                ),
            )

    by_node: dict[
        str,
        list[tuple[float, tuple[str, ...]]],
    ] = {}

    for (
        node,
        _hops,
        _mechanism,
        _alignment,
    ), value in best_semantic_state.items():
        by_node.setdefault(
            node,
            [],
        ).append(
            value
        )

    return {
        node: tuple(
            sorted(
                set(values),
                key=lambda item: (
                    len(item[1]) - 1,
                    item[0],
                    item[1],
                ),
            )
        )
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
    context_switch_penalty: float
    path_length_penalty: float
    total: float

    @property
    def reaction_switch_penalty(self) -> float:
        # Deprecated v2.8 compatibility alias.
        return self.context_switch_penalty

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reaction_switch_penalty"] = self.context_switch_penalty
        return payload


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
    context_node_labels: tuple[str, ...]
    visited_paper_ids: tuple[str, ...]

    @property
    def reaction_node_labels(self) -> tuple[str, ...]:
        # Deprecated v2.8 compatibility alias.
        return self.context_node_labels

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
            "context_node_labels": list(self.context_node_labels),
            "reaction_node_labels": list(self.context_node_labels),
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

    # RCF-7 correction contract.
    #
    # False preserves the historical selector behavior.
    # True atomically enables:
    #   - semantic-state path retention,
    #   - score-first route choice within one endpoint/anchor quadruple,
    #   - alignment-hub provenance exclusion,
    #   - narrow routing-carrier reuse with candidate-anchor reuse forbidden.
    corrected_route_contract: bool = False

    endpoint_weight: float = 0.18
    unit_relevance_weight: float = 0.24
    mechanistic_continuity_weight: float = 0.24
    scientific_density_weight: float = 0.16
    cross_paper_weight: float = 0.08

    generic_penalty_weight: float = 0.16
    alignment_penalty_weight: float = 0.10
    reverse_penalty_weight: float = 0.05
    context_switch_penalty_weight: float = 0.18
    path_length_penalty_weight: float = 0.08

    @property
    def reaction_switch_penalty_weight(self) -> float:
        # Deprecated v2.8 compatibility alias.
        return self.context_switch_penalty_weight

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reaction_switch_penalty_weight"] = self.context_switch_penalty_weight
        return payload


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
        domain_profile: ScientificDomainProfile,
    ) -> None:
        self.graph = graph
        self.confirmed_graph = confirmed_graph
        self.policy = policy or CandidateUnitSelectionPolicy()
        self.domain_profile = domain_profile
        self.discovery_semantics = self.domain_profile.discovery
        self.unit_relevance = {str(k): _clip01(float(v)) for k, v in dict(unit_relevance or {}).items()}
        self.unit_vectors = dict(unit_vectors or {})
        self._corrected_reusable_carriers = (
            self._build_corrected_reusable_carriers()
            if self.policy.corrected_route_contract
            else frozenset()
        )

    def _build_corrected_reusable_carriers(
        self,
    ) -> frozenset[str]:
        """Return the narrow routing-only carrier class validated by RCF.

        Reusable nodes are:
        1. corpus-alignment nodes; or
        2. shared scientific entities whose incident confirmed edges are
           exclusively domain scaffold relations or corpus-alignment edges.

        Candidate anchors are *not* authorized here.  They are removed again
        per candidate unit by ``_route_reuse_allowed``.
        """
        reusable: set[str] = set()

        for raw_node in self.confirmed_graph.nodes:
            node = str(raw_node)

            attrs = dict(
                self.confirmed_graph.nodes[node]
            )

            if is_alignment_node(attrs):
                reusable.add(node)
                continue

            if not is_shared_entity_node(
                node,
                attrs,
                self.discovery_semantics,
            ):
                continue

            incident = [
                dict(edge)
                for _, _, edge
                in self.confirmed_graph.in_edges(
                    node,
                    data=True,
                )
            ]

            incident.extend(
                dict(edge)
                for _, _, edge
                in self.confirmed_graph.out_edges(
                    node,
                    data=True,
                )
            )

            if (
                incident
                and all(
                    is_scaffold_edge(
                        edge,
                        self.discovery_semantics,
                    )
                    or is_semantic_alignment_edge(
                        edge
                    )
                    for edge in incident
                )
            ):
                reusable.add(node)

        return frozenset(
            reusable
        )

    def _route_reuse_allowed(
        self,
        nodes: tuple[str, ...],
        unit: CandidateUnit,
    ) -> bool:
        seen: set[str] = set()
        duplicates: set[str] = set()

        for node in nodes:
            if node in seen:
                duplicates.add(node)
            else:
                seen.add(node)

        if not duplicates:
            return True

        if not self.policy.corrected_route_contract:
            return False

        strict_anchors = {
            anchor.node_id
            for anchor in unit.anchors
        }

        for node in duplicates:
            if node in strict_anchors:
                return False

            if (
                node
                not in self._corrected_reusable_carriers
            ):
                return False

        return True

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
        left_mech = any(is_mechanism_edge(step, self.discovery_semantics) for step in left_edges) or any(
            node in self.graph and is_mechanism_node(node, dict(self.graph.nodes[node]), self.discovery_semantics)
            for node in left_nodes
        )
        right_mech = any(is_mechanism_edge(step, self.discovery_semantics) for step in right_edges) or any(
            node in self.graph and is_mechanism_node(
                node,
                dict(self.graph.nodes[node]),
                self.discovery_semantics,
            )
            for node in right_nodes
        )
        mechanistic_continuity = 1.0 if left_mech and right_mech else (0.55 if left_mech or right_mech else 0.0)

        generic_flags = []
        for node in confirmed_nodes[1:-1]:
            if node not in self.graph:
                continue
            attrs = dict(self.graph.nodes[node])
            if is_alignment_node(attrs):
                continue
            generic_flags.append(is_generic_entity_node(node, attrs, self.discovery_semantics))
        generic_penalty = _clip01(sum(generic_flags) / len(generic_flags)) if generic_flags else 0.0

        alignment_count = sum(edge_is_alignment(step) for step in confirmed_steps)
        alignment_penalty = _clip01(alignment_count / len(confirmed_steps)) if confirmed_steps else 0.0
        reverse_count = sum(edge_is_reverse(step) for step in confirmed_steps)
        reverse_penalty = _clip01(reverse_count / len(confirmed_steps)) if confirmed_steps else 0.0

        scaffold_count = sum(is_scaffold_edge(step, self.discovery_semantics) or edge_is_alignment(step) for step in confirmed_steps)
        scientific_density = _clip01(1.0 - scaffold_count / len(confirmed_steps)) if confirmed_steps else 0.0

        # Context dimensions are typed. Multiple complementary context types
        # (for example SERS Analyte + RamanReporter + OpticalCondition) do not
        # constitute a switch. A switch occurs only when a route traverses more
        # than one distinct value *within the same configured context type*.
        context_labels_by_type: dict[str, set[str]] = {}
        context_types = self.discovery_semantics.normalized_context_node_types()
        for node in confirmed_nodes:
            if node not in self.graph:
                continue
            attrs = dict(self.graph.nodes[node])
            node_type = normalized_node_type(attrs)
            if node_type in context_types:
                context_labels_by_type.setdefault(node_type, set()).add(
                    node_label(self.graph, node)
                )

        unique_contexts = tuple(
            sorted(
                label
                for labels in context_labels_by_type.values()
                for label in labels
            )
        )
        context_switch_count = sum(
            max(0, len(labels) - 1)
            for labels in context_labels_by_type.values()
        )
        context_switch_penalty = _clip01(context_switch_count / 2.0)

        papers: set[str] = set()
        for node in nodes:
            if (
                self.policy.corrected_route_contract
                and node in self.graph
                and is_alignment_node(
                    dict(self.graph.nodes[node])
                )
            ):
                # Corpus-alignment hubs can summarize hundreds of papers.
                # They are routing aids, not scientific evidence visited by
                # this candidate route.
                continue

            papers.update(
                paper_ids_from_node(
                    self.graph,
                    node,
                )
            )

        cross_paper_span = _clip01((len(papers) - 1) / 2.0) if papers else 0.0
        return (
            mechanistic_continuity,
            scientific_density,
            generic_penalty,
            alignment_penalty,
            reverse_penalty,
            unique_contexts,
            context_switch_penalty,
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
            context_labels,
            context_penalty,
            papers,
        ) = self._route_diagnostics(
            nodes,
            candidate_id=unit.candidate_node_id,
            entry_id=entry_id,
            exit_id=exit_id,
        )
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
            - p.context_switch_penalty_weight * context_penalty
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
                context_switch_penalty=context_penalty,
                path_length_penalty=length_penalty,
                total=total,
            ),
            context_labels,
            papers,
        )

    def _enumerate_routes_corrected(
        self,
        units: list[CandidateUnit],
        sources: list[dict[str, Any]],
        targets: list[dict[str, Any]],
    ) -> list[CandidateUnitRoute]:
        side_hops = self.policy.max_depth - 2

        prefix_cache = {
            str(match["node_id"]): _bounded_shortest_paths(
                self.confirmed_graph,
                str(match["node_id"]),
                max_hops=side_hops,
                discovery_semantics=self.discovery_semantics,
                retain_semantic_state=True,
            )
            for match in sources
        }

        reverse_confirmed = self.confirmed_graph.reverse(
            copy=False
        )

        suffix_cache = {
            str(match["node_id"]): _bounded_shortest_paths(
                reverse_confirmed,
                str(match["node_id"]),
                max_hops=side_hops,
                discovery_semantics=self.discovery_semantics,
                retain_semantic_state=True,
            )
            for match in targets
        }

        routes: list[CandidateUnitRoute] = []

        for unit in units:
            relevance = self.unit_relevance.get(
                unit.candidate_node_id,
                0.0,
            )

            if (
                relevance
                < self.policy.min_unit_relevance
            ):
                continue

            candidate_id = (
                unit.candidate_node_id
            )

            for entry in unit.anchors:
                if not self.graph.has_edge(
                    entry.node_id,
                    candidate_id,
                ):
                    continue

                first_candidate = dict(
                    self.graph.edges[
                        entry.node_id,
                        candidate_id,
                    ]
                )

                if (
                    not edge_is_candidate(
                        first_candidate
                    )
                    or edge_is_reverse(
                        first_candidate
                    )
                ):
                    continue

                for exit_anchor in unit.anchors:
                    if (
                        entry.node_id
                        == exit_anchor.node_id
                    ):
                        continue

                    if not self.graph.has_edge(
                        candidate_id,
                        exit_anchor.node_id,
                    ):
                        continue

                    second_candidate = dict(
                        self.graph.edges[
                            candidate_id,
                            exit_anchor.node_id,
                        ]
                    )

                    if (
                        not edge_is_candidate(
                            second_candidate
                        )
                        or not edge_is_reverse(
                            second_candidate
                        )
                    ):
                        continue

                    for source_match in sources:
                        source_id = str(
                            source_match[
                                "node_id"
                            ]
                        )

                        prefix_variants = (
                            prefix_cache[
                                source_id
                            ].get(
                                entry.node_id,
                                (),
                            )
                        )

                        if not prefix_variants:
                            continue

                        for target_match in targets:
                            target_id = str(
                                target_match[
                                    "node_id"
                                ]
                            )

                            suffix_variants = (
                                suffix_cache[
                                    target_id
                                ].get(
                                    exit_anchor.node_id,
                                    (),
                                )
                            )

                            if not suffix_variants:
                                continue

                            best_route: (
                                CandidateUnitRoute
                                | None
                            ) = None

                            best_key: (
                                tuple[Any, ...]
                                | None
                            ) = None

                            for (
                                prefix_cost,
                                prefix_nodes,
                            ) in prefix_variants:
                                prefix_hops = (
                                    len(prefix_nodes)
                                    - 1
                                )

                                for (
                                    suffix_cost,
                                    suffix_reverse_nodes,
                                ) in suffix_variants:
                                    suffix_hops = (
                                        len(
                                            suffix_reverse_nodes
                                        )
                                        - 1
                                    )

                                    hop_count = (
                                        prefix_hops
                                        + 2
                                        + suffix_hops
                                    )

                                    if (
                                        hop_count
                                        > self.policy.max_depth
                                    ):
                                        continue

                                    suffix_nodes = tuple(
                                        reversed(
                                            suffix_reverse_nodes
                                        )
                                    )

                                    nodes = (
                                        prefix_nodes
                                        + (
                                            candidate_id,
                                        )
                                        + suffix_nodes
                                    )

                                    if not self._route_reuse_allowed(
                                        nodes,
                                        unit,
                                    ):
                                        continue

                                    total_cost = (
                                        float(prefix_cost)
                                        + float(
                                            first_candidate.get(
                                                "exploration_cost",
                                                1.0,
                                            )
                                        )
                                        + float(
                                            second_candidate.get(
                                                "exploration_cost",
                                                1.0,
                                            )
                                        )
                                        + float(suffix_cost)
                                    )

                                    (
                                        score,
                                        contexts,
                                        papers,
                                    ) = self._score(
                                        source_match=
                                            source_match,
                                        target_match=
                                            target_match,
                                        unit=unit,
                                        nodes=nodes,
                                        entry_id=
                                            entry.node_id,
                                        exit_id=
                                            exit_anchor.node_id,
                                    )

                                    if (
                                        score.total
                                        < self.policy.min_selection_score
                                    ):
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

                                    candidate_route = (
                                        CandidateUnitRoute(
                                            route_id=
                                                route_id,
                                            unit=unit,
                                            entry_anchor=
                                                entry,
                                            exit_anchor=
                                                exit_anchor,
                                            source_match=
                                                source_match,
                                            target_match=
                                                target_match,
                                            nodes=nodes,
                                            total_cost=
                                                total_cost,
                                            hop_count=
                                                hop_count,
                                            score=score,
                                            context_node_labels=
                                                contexts,
                                            visited_paper_ids=
                                                papers,
                                        )
                                    )

                                    # Scientific score is the local objective.
                                    # Cost is only a deterministic tie-breaker.
                                    candidate_key = (
                                        -score.total,
                                        -score.unit_relevance,
                                        score.context_switch_penalty,
                                        total_cost,
                                        hop_count,
                                        route_id,
                                    )

                                    if (
                                        best_key is None
                                        or candidate_key
                                        < best_key
                                    ):
                                        best_key = (
                                            candidate_key
                                        )

                                        best_route = (
                                            candidate_route
                                        )

                            if best_route is not None:
                                routes.append(
                                    best_route
                                )

        routes.sort(
            key=lambda route: (
                -route.score.total,
                -route.score.unit_relevance,
                route.score.context_switch_penalty,
                route.total_cost,
                route.hop_count,
                route.route_id,
            )
        )

        return routes

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

        if self.policy.corrected_route_contract:
            return self._enumerate_routes_corrected(
                units,
                sources,
                targets,
            )

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
                            score, contexts, papers = self._score(
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
                                    context_node_labels=contexts,
                                    visited_paper_ids=papers,
                                )
                            )
        routes.sort(
            key=lambda route: (
                -route.score.total,
                -route.score.unit_relevance,
                route.score.context_switch_penalty,
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
