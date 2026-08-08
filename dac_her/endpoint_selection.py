from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import networkx as nx


def _as_float(
    value: Any,
    default: float = 0.0,
) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _component_index(
    graph: nx.DiGraph,
) -> tuple[
    dict[str, int],
    dict[int, int],
]:
    components = [
        sorted(map(str, component))
        for component
        in nx.weakly_connected_components(graph)
    ]
    components.sort(
        key=lambda nodes: (
            -len(nodes),
            nodes[0] if nodes else "",
        )
    )

    node_to_component: dict[str, int] = {}
    component_sizes: dict[int, int] = {}

    for component_id, nodes in enumerate(
        components
    ):
        component_sizes[component_id] = len(
            nodes
        )
        for node_id in nodes:
            node_to_component[node_id] = (
                component_id
            )

    return (
        node_to_component,
        component_sizes,
    )


@dataclass(frozen=True)
class EndpointPairDiagnostic:
    source_node_id: str
    target_node_id: str
    source_label: str
    target_label: str
    source_similarity: float
    target_similarity: float
    source_exact: bool
    target_exact: bool
    semantic_tier: int
    source_component_id: int | None
    target_component_id: int | None
    source_component_size: int
    target_component_size: int
    same_weak_component: bool
    directed_reachable: bool
    shortest_hops: int | None
    shortest_weighted_cost: float | None
    within_max_depth: bool
    pair_score: float
    selected: bool
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EndpointPairSelector:
    """Rank semantic endpoints without letting reachability replace meaning.

    Exact-label semantic tiers are primary. Reachability, hop count and
    weighted navigation cost refine selection inside those tiers. Concrete
    path validity remains the TraversalEngine's responsibility.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        exact_match_bonus: float = 0.12,
        containment_bonus: float = 0.02,
        hop_penalty: float = 0.015,
        cost_penalty: float = 0.005,
    ) -> None:
        if graph.is_multigraph():
            raise TypeError(
                "EndpointPairSelector requires "
                "the collapsed NavigationGraph."
            )
        if not graph.is_directed():
            raise TypeError(
                "EndpointPairSelector requires "
                "a directed graph."
            )

        self.graph = graph
        self.exact_match_bonus = (
            exact_match_bonus
        )
        self.containment_bonus = (
            containment_bonus
        )
        self.hop_penalty = hop_penalty
        self.cost_penalty = cost_penalty
        (
            self.node_to_component,
            self.component_sizes,
        ) = _component_index(graph)

    @staticmethod
    def _semantic_tier(
        source_exact: bool,
        target_exact: bool,
    ) -> int:
        if source_exact and target_exact:
            return 0
        if source_exact or target_exact:
            return 1
        return 2

    def _single_source(
        self,
        source: str,
    ) -> tuple[
        dict[str, int],
        dict[str, float],
    ]:
        if source not in self.graph:
            return {}, {}

        hops = {
            str(node_id): int(distance)
            for node_id, distance
            in nx.single_source_shortest_path_length(
                self.graph,
                source,
            ).items()
        }
        costs = {
            str(node_id): float(distance)
            for node_id, distance
            in nx.single_source_dijkstra_path_length(
                self.graph,
                source,
                weight="exploration_cost",
            ).items()
        }
        return hops, costs

    def select(
        self,
        source_matches: Iterable[
            dict[str, Any]
        ],
        target_matches: Iterable[
            dict[str, Any]
        ],
        *,
        top_k: int = 12,
        max_depth: int = 8,
    ) -> tuple[
        list[EndpointPairDiagnostic],
        list[EndpointPairDiagnostic],
    ]:
        if top_k <= 0:
            return [], []

        sources = [
            dict(item)
            for item in source_matches
        ]
        targets = [
            dict(item)
            for item in target_matches
        ]

        diagnostics: list[
            EndpointPairDiagnostic
        ] = []
        source_cache: dict[
            str,
            tuple[
                dict[str, int],
                dict[str, float],
            ],
        ] = {}

        for source_match in sources:
            source_id = str(
                source_match["node_id"]
            )
            source_label = str(
                source_match.get("label")
                or source_id
            )
            source_similarity = _as_float(
                source_match.get(
                    "semantic_similarity"
                ),
                (
                    1.0
                    if source_match.get(
                        "direct_node_id"
                    )
                    else 0.0
                ),
            )
            source_exact = (
                _as_bool(
                    source_match.get(
                        "exact_label_match",
                        False,
                    )
                )
                or bool(
                    source_match.get(
                        "direct_node_id"
                    )
                )
            )
            source_contains = _as_bool(
                source_match.get(
                    "label_contains_query",
                    False,
                )
            )
            source_component = (
                self.node_to_component.get(
                    source_id
                )
            )
            source_component_size = (
                self.component_sizes.get(
                    source_component,
                    0,
                )
                if source_component is not None
                else 0
            )

            if source_id not in source_cache:
                source_cache[source_id] = (
                    self._single_source(
                        source_id
                    )
                )
            (
                hop_lengths,
                weighted_costs,
            ) = source_cache[source_id]

            for target_match in targets:
                target_id = str(
                    target_match["node_id"]
                )
                target_label = str(
                    target_match.get("label")
                    or target_id
                )
                target_similarity = _as_float(
                    target_match.get(
                        "semantic_similarity"
                    ),
                    (
                        1.0
                        if target_match.get(
                            "direct_node_id"
                        )
                        else 0.0
                    ),
                )
                target_exact = (
                    _as_bool(
                        target_match.get(
                            "exact_label_match",
                            False,
                        )
                    )
                    or bool(
                        target_match.get(
                            "direct_node_id"
                        )
                    )
                )
                target_contains = _as_bool(
                    target_match.get(
                        "label_contains_query",
                        False,
                    )
                )

                target_component = (
                    self.node_to_component.get(
                        target_id
                    )
                )
                target_component_size = (
                    self.component_sizes.get(
                        target_component,
                        0,
                    )
                    if target_component is not None
                    else 0
                )

                same_component = (
                    source_component is not None
                    and target_component is not None
                    and source_component
                    == target_component
                )
                directed_reachable = (
                    target_id
                    in hop_lengths
                )
                shortest_hops = (
                    hop_lengths.get(
                        target_id
                    )
                )
                weighted_cost = (
                    weighted_costs.get(
                        target_id
                    )
                )
                within_depth = (
                    directed_reachable
                    and shortest_hops is not None
                    and shortest_hops
                    <= max_depth
                )

                semantic_tier = (
                    self._semantic_tier(
                        source_exact,
                        target_exact,
                    )
                )
                semantic_score = (
                    source_similarity
                    + target_similarity
                ) / 2.0

                pair_score = semantic_score
                pair_score += (
                    self.exact_match_bonus
                    * (
                        int(source_exact)
                        + int(target_exact)
                    )
                )
                pair_score += (
                    self.containment_bonus
                    * (
                        int(source_contains)
                        + int(target_contains)
                    )
                )
                if shortest_hops is not None:
                    pair_score -= (
                        self.hop_penalty
                        * shortest_hops
                    )
                if weighted_cost is not None:
                    pair_score -= (
                        self.cost_penalty
                        * weighted_cost
                    )

                if not same_component:
                    reason = (
                        "different_weak_component"
                    )
                elif not directed_reachable:
                    reason = (
                        "not_directed_reachable"
                    )
                elif not within_depth:
                    reason = (
                        "reachable_beyond_max_depth"
                    )
                else:
                    reason = "eligible"

                diagnostics.append(
                    EndpointPairDiagnostic(
                        source_node_id=source_id,
                        target_node_id=target_id,
                        source_label=source_label,
                        target_label=target_label,
                        source_similarity=(
                            source_similarity
                        ),
                        target_similarity=(
                            target_similarity
                        ),
                        source_exact=source_exact,
                        target_exact=target_exact,
                        semantic_tier=(
                            semantic_tier
                        ),
                        source_component_id=(
                            source_component
                        ),
                        target_component_id=(
                            target_component
                        ),
                        source_component_size=(
                            source_component_size
                        ),
                        target_component_size=(
                            target_component_size
                        ),
                        same_weak_component=(
                            same_component
                        ),
                        directed_reachable=(
                            directed_reachable
                        ),
                        shortest_hops=(
                            shortest_hops
                        ),
                        shortest_weighted_cost=(
                            weighted_cost
                        ),
                        within_max_depth=(
                            within_depth
                        ),
                        pair_score=float(
                            pair_score
                        ),
                        selected=False,
                        selection_reason=reason,
                    )
                )

        eligible = [
            item
            for item in diagnostics
            if item.selection_reason
            == "eligible"
        ]

        eligible.sort(
            key=lambda item: (
                item.semantic_tier,
                -item.pair_score,
                -(
                    item.source_similarity
                    + item.target_similarity
                ),
                (
                    item.shortest_hops
                    if item.shortest_hops
                    is not None
                    else 10**9
                ),
                (
                    item.shortest_weighted_cost
                    if item.shortest_weighted_cost
                    is not None
                    else float("inf")
                ),
                item.source_node_id,
                item.target_node_id,
            )
        )

        selected_keys = {
            (
                item.source_node_id,
                item.target_node_id,
            )
            for item
            in eligible[:top_k]
        }

        selected: list[
            EndpointPairDiagnostic
        ] = []
        updated: list[
            EndpointPairDiagnostic
        ] = []

        for item in diagnostics:
            key = (
                item.source_node_id,
                item.target_node_id,
            )
            if key in selected_keys:
                payload = item.to_dict()
                payload["selected"] = True
                payload[
                    "selection_reason"
                ] = "selected_reachable_pair"
                chosen = (
                    EndpointPairDiagnostic(
                        **payload
                    )
                )
                selected.append(chosen)
                updated.append(chosen)
            else:
                updated.append(item)

        selected.sort(
            key=lambda item: (
                item.semantic_tier,
                -item.pair_score,
                (
                    item.shortest_hops
                    if item.shortest_hops
                    is not None
                    else 10**9
                ),
                item.source_node_id,
                item.target_node_id,
            )
        )

        updated.sort(
            key=lambda item: (
                0 if item.selected else 1,
                item.semantic_tier,
                (
                    0
                    if item.selection_reason
                    == "reachable_beyond_max_depth"
                    else 1
                ),
                -item.pair_score,
                item.source_node_id,
                item.target_node_id,
            )
        )

        return selected, updated
