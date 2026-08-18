from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import networkx as nx


SemanticRoleNormalizer = Callable[..., tuple[nx.MultiDiGraph, list[Any]]]


@dataclass(frozen=True)
class RelationConstraint:
    """Allowed endpoint types for one paper-graph relation."""

    relation: str
    source_types: frozenset[str] = frozenset()
    target_types: frozenset[str] = frozenset()
    severity: str = "warning"

    def check(
        self,
        *,
        source_id: str,
        source_type: str | None,
        target_id: str,
        target_type: str | None,
        edge_key: str,
    ) -> list["GraphSemanticIssue"]:
        issues: list[GraphSemanticIssue] = []
        if self.source_types and source_type not in self.source_types:
            issues.append(
                GraphSemanticIssue(
                    severity=self.severity,
                    code="relation_source_type_mismatch",
                    relation=self.relation,
                    source_id=source_id,
                    source_type=source_type or "",
                    target_id=target_id,
                    target_type=target_type or "",
                    edge_key=edge_key,
                    expected=sorted(self.source_types),
                    message=(
                        f"{self.relation} source {source_id!r} has type "
                        f"{source_type!r}; expected one of "
                        f"{sorted(self.source_types)!r}."
                    ),
                )
            )
        if self.target_types and target_type not in self.target_types:
            issues.append(
                GraphSemanticIssue(
                    severity=self.severity,
                    code="relation_target_type_mismatch",
                    relation=self.relation,
                    source_id=source_id,
                    source_type=source_type or "",
                    target_id=target_id,
                    target_type=target_type or "",
                    edge_key=edge_key,
                    expected=sorted(self.target_types),
                    message=(
                        f"{self.relation} target {target_id!r} has type "
                        f"{target_type!r}; expected one of "
                        f"{sorted(self.target_types)!r}."
                    ),
                )
            )
        return issues


@dataclass(frozen=True)
class GraphSemanticIssue:
    severity: str
    code: str
    relation: str
    source_id: str
    source_type: str
    target_id: str
    target_type: str
    edge_key: str
    expected: list[str]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphDomainAdapter:
    """Domain-owned paper-graph semantics."""

    adapter_id: str
    domain_profile_id: str
    semantic_role_policy: str
    semantic_role_normalizer: SemanticRoleNormalizer
    relation_constraints: tuple[RelationConstraint, ...] = ()

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

    def diagnose_relation_contracts(
        self,
        graph: nx.MultiDiGraph,
    ) -> list[GraphSemanticIssue]:
        by_relation = {
            constraint.relation: constraint
            for constraint in self.relation_constraints
        }
        issues: list[GraphSemanticIssue] = []
        for source, target, key, data in graph.edges(keys=True, data=True):
            relation = str(data.get("relation", ""))
            constraint = by_relation.get(relation)
            if constraint is None:
                continue
            source_type = graph.nodes[source].get("type")
            target_type = graph.nodes[target].get("type")
            issues.extend(
                constraint.check(
                    source_id=str(source),
                    source_type=str(source_type) if source_type is not None else None,
                    target_id=str(target),
                    target_type=str(target_type) if target_type is not None else None,
                    edge_key=str(key),
                )
            )
        return issues
