from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
)

import networkx as nx


_ALLOWED_TRUE_VALUES = {
    "1",
    "true",
    "yes",
}


def _as_bool(
    value: object,
) -> bool:
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in _ALLOWED_TRUE_VALUES
    )


def candidate_edge_count(
    graph: nx.DiGraph,
    path: Iterable[str],
) -> int:
    nodes = list(path)

    return sum(
        _as_bool(
            graph.edges[
                left,
                right,
            ].get(
                "requires_verification",
                False,
            )
        )
        for left, right in zip(
            nodes,
            nodes[1:],
            strict=False,
        )
    )


def path_allowed(
    graph: nx.DiGraph,
    path: Iterable[str],
    *,
    mode: str,
) -> bool:
    count = candidate_edge_count(
        graph,
        path,
    )

    if mode == "mechanism":
        return count == 0

    if mode == "exploratory":
        return count <= 1

    if mode == "evidence":
        return count == 0

    raise ValueError(
        f"Unknown traversal mode: "
        f"{mode!r}"
    )


def path_cost(
    graph: nx.DiGraph,
    path: Iterable[str],
) -> float:
    nodes = list(path)

    return sum(
        float(
            graph.edges[
                left,
                right,
            ].get(
                "exploration_cost",
                1.0,
            )
        )
        for left, right in zip(
            nodes,
            nodes[1:],
            strict=False,
        )
    )


def iter_allowed_shortest_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
    *,
    mode: str,
    top_k: int = 10,
) -> Iterator[list[str]]:
    if top_k <= 0:
        return

    generated = 0

    raw_paths = (
        nx.shortest_simple_paths(
            graph,
            source,
            target,
            weight="exploration_cost",
        )
    )

    for path in raw_paths:
        normalized_path = [
            str(node_id)
            for node_id in path
        ]

        if not path_allowed(
            graph,
            normalized_path,
            mode=mode,
        ):
            continue

        yield normalized_path

        generated += 1

        if generated >= top_k:
            break