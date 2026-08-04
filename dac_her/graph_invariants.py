from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import networkx as nx


PAPER_NODE_TYPE = "Paper"

SUBJECT_TYPES = {
    "Catalyst",
    "CatalystModel",
    "Material",
    "Support",
}

COMPONENT_ANCHOR_PRIORITY = {
    "Catalyst": 0,
    "CatalystModel": 1,
    "Material": 2,
    "Support": 3,
    "Experiment": 4,
    "Calculation": 5,
    "ObservationClaim": 6,
    "MechanismClaim": 7,
}


def _stable_key(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ensure_multidigraph(graph: nx.Graph) -> nx.MultiDiGraph:
    """
    Return an equivalent MultiDiGraph.

    This function is idempotent and preserves graph, node, and edge
    attributes.
    """
    if isinstance(graph, nx.MultiDiGraph):
        return graph

    converted = nx.MultiDiGraph()
    converted.graph.update(graph.graph)
    converted.add_nodes_from(graph.nodes(data=True))

    if graph.is_multigraph():
        for source, target, key, attrs in graph.edges(
            keys=True,
            data=True,
        ):
            converted.add_edge(
                source,
                target,
                key=str(key),
                **dict(attrs),
            )
    else:
        for index, (source, target, attrs) in enumerate(
            graph.edges(data=True)
        ):
            attrs = dict(attrs)

            key = str(
                attrs.get("edge_key")
                or attrs.get("id")
                or _stable_key(source, target, index)
            )

            while converted.has_edge(source, target, key):
                key = f"{key}_{index}"

            converted.add_edge(
                source,
                target,
                key=key,
                **attrs,
            )

    return converted