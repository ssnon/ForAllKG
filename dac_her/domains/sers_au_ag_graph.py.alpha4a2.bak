from __future__ import annotations

import networkx as nx

from dac_her.graph_domain import GraphDomainAdapter


def _preserve_sers_semantic_roles(
    graph: nx.MultiDiGraph,
    *,
    chunk_id: str,
):
    """Preserve strict SERS roles; do not infer electrocatalyst roles."""
    del chunk_id
    return graph, []


SERS_AU_AG_GRAPH_ADAPTER = GraphDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    semantic_role_policy=(
        "No electrocatalysis-specific semantic-role coercion is applied. "
        "Strict SERS entity types are preserved through paper-graph merging."
    ),
    semantic_role_normalizer=_preserve_sers_semantic_roles,
)
