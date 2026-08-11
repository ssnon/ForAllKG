from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx


SemanticRoleNormalizer = Callable[..., tuple[nx.MultiDiGraph, list[Any]]]


@dataclass(frozen=True)
class GraphDomainAdapter:
    """Domain-owned paper-graph semantics."""

    adapter_id: str
    domain_profile_id: str
    semantic_role_policy: str
    semantic_role_normalizer: SemanticRoleNormalizer

    def normalize_semantic_roles(
        self,
        graph: nx.MultiDiGraph,
        *,
        chunk_id: str,
    ) -> tuple[nx.MultiDiGraph, list[Any]]:
        return self.semantic_role_normalizer(
            graph,
            chunk_id=chunk_id,
        )
