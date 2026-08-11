from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from dac_her.node_references import remap_node_reference_attributes
from dac_her.resolution_candidates import (
    CLAIM_NODE_TYPES,
    RESOLVABLE_NODE_TYPES,
    read_jsonl,
)


@dataclass(frozen=True)
class ResolutionPlan:
    aliases: dict[str, str]
    drop_node_ids: set[str]
    source_path: str | None
    source_format: str
    approved_same_entity: int
    applied_aliases: int
    stale_decisions: int
    ignored_decisions: int
    rejected_decisions: int
    cluster_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_format": self.source_format,
            "approved_same_entity": self.approved_same_entity,
            "applied_aliases": self.applied_aliases,
            "stale_decisions": self.stale_decisions,
            "ignored_decisions": self.ignored_decisions,
            "rejected_decisions": self.rejected_decisions,
            "cluster_count": self.cluster_count,
        }


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_rank = self.rank[left_root]
        right_rank = self.rank[right_root]
        if left_rank < right_rank:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if left_rank == right_rank:
            self.rank[left_root] += 1

    def groups(self) -> dict[str, set[str]]:
        groups: dict[str, set[str]] = defaultdict(set)
        for value in self.parent:
            groups[self.find(value)].add(value)
        return groups


def _load_legacy_resolution_json(
    path: Path,
) -> ResolutionPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            "Legacy resolution decision file must contain a JSON object."
        )

    raw_aliases = payload.get("aliases", {})
    raw_drop = payload.get("drop_node_ids", [])

    if not isinstance(raw_aliases, dict):
        raise ValueError("'aliases' must be a JSON object.")
    if not isinstance(raw_drop, list):
        raise ValueError("'drop_node_ids' must be a JSON array.")

    aliases: dict[str, str] = {}
    for source, target in raw_aliases.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Alias IDs must be strings.")
        if source == target:
            continue
        aliases[source] = target

    drop_node_ids = {str(node_id) for node_id in raw_drop}
    return ResolutionPlan(
        aliases=aliases,
        drop_node_ids=drop_node_ids,
        source_path=str(path),
        source_format="legacy_json",
        approved_same_entity=len(aliases),
        applied_aliases=len(aliases),
        stale_decisions=0,
        ignored_decisions=0,
        rejected_decisions=0,
        cluster_count=len(set(aliases.values())),
    )


def _is_blank(value: Any) -> bool:
    return value is None or value == ""


def _representative_score(
    graph: nx.Graph,
    node_id: str,
) -> tuple[int, int, int, str]:
    data = graph.nodes[node_id]
    nonempty = sum(
        not _is_blank(value)
        for value in data.values()
    )
    degree = int(graph.degree(node_id))
    label = str(data.get("label", ""))
    return (nonempty, degree, -len(label), node_id)


def _choose_representative(
    graph: nx.Graph,
    members: set[str],
    explicit_choices: set[str],
) -> str:
    valid_explicit = explicit_choices & members
    if len(valid_explicit) > 1:
        raise ValueError(
            "Conflicting canonical_id choices for one resolution cluster: "
            f"{sorted(valid_explicit)}"
        )
    if valid_explicit:
        return next(iter(valid_explicit))
    return max(
        sorted(members),
        key=lambda node_id: _representative_score(graph, node_id),
    )


def _load_jsonl_resolution_plan(
    path: Path,
    *,
    graph: nx.Graph,
) -> ResolutionPlan:
    records = read_jsonl(path)
    dsu = _DisjointSet()
    approved_records: list[tuple[str, str, str | None, bool]] = []

    approved_same_entity = 0
    stale_decisions = 0
    ignored_decisions = 0
    rejected_decisions = 0

    for index, record in enumerate(records, start=1):
        decision = str(record.get("decision", "unreviewed")).strip()
        approved = bool(record.get("approved", False))

        if decision != "same_entity" or not approved:
            ignored_decisions += 1
            continue

        approved_same_entity += 1
        left_id = str(record.get("left_id", "")).strip()
        right_id = str(record.get("right_id", "")).strip()
        if not left_id or not right_id or left_id == right_id:
            rejected_decisions += 1
            continue

        if left_id not in graph or right_id not in graph:
            stale_decisions += 1
            continue

        left_type = str(graph.nodes[left_id].get("type", ""))
        right_type = str(graph.nodes[right_id].get("type", ""))
        if left_type != right_type:
            raise ValueError(
                f"Approved same_entity decision #{index} has different node "
                f"types: {left_id!r} ({left_type}) vs "
                f"{right_id!r} ({right_type})."
            )
        if left_type in CLAIM_NODE_TYPES:
            raise ValueError(
                "Claim nodes are not eligible for destructive paper-level "
                f"same_entity merging: {left_id!r}, {right_id!r}."
            )
        if left_type not in RESOLVABLE_NODE_TYPES and left_type != "Measurement":
            raise ValueError(
                f"Unsupported node type in resolution decision: {left_type!r}."
            )

        canonical_id_value = record.get("canonical_id")
        canonical_id = (
            str(canonical_id_value).strip()
            if not _is_blank(canonical_id_value)
            else None
        )
        reviewer = str(record.get("reviewer", "") or "").strip()
        automatic_choice = reviewer == "automatic_registry_rule"

        dsu.union(left_id, right_id)
        approved_records.append(
            (left_id, right_id, canonical_id, automatic_choice)
        )

    groups = dsu.groups()
    explicit_by_root: dict[str, set[str]] = defaultdict(set)
    for left_id, right_id, canonical_id, automatic_choice in approved_records:
        if canonical_id is None or automatic_choice:
            # Pairwise auto-approved candidates can form a transitive cluster.
            # Their old left-node canonical hints are not cluster-level choices
            # and must not conflict with one another. The representative is
            # selected deterministically after the full cluster is known.
            continue
        root = dsu.find(left_id)
        members = groups[root]
        if canonical_id not in members:
            raise ValueError(
                "Human canonical_id choice is outside its resolution cluster: "
                f"canonical_id={canonical_id!r}, pair=({left_id!r}, "
                f"{right_id!r}), cluster={sorted(members)!r}."
            )
        explicit_by_root[root].add(canonical_id)

    aliases: dict[str, str] = {}
    cluster_count = 0
    for root, members in groups.items():
        if len(members) < 2:
            continue
        cluster_count += 1
        representative = _choose_representative(
            graph,
            members,
            explicit_by_root.get(root, set()),
        )
        for member in sorted(members):
            if member != representative:
                aliases[member] = representative

    return ResolutionPlan(
        aliases=aliases,
        drop_node_ids=set(),
        source_path=str(path),
        source_format="jsonl_reviewed_decisions",
        approved_same_entity=approved_same_entity,
        applied_aliases=len(aliases),
        stale_decisions=stale_decisions,
        ignored_decisions=ignored_decisions,
        rejected_decisions=rejected_decisions,
        cluster_count=cluster_count,
    )


def load_resolution_plan(
    path: str | Path | None,
    *,
    graph: nx.Graph,
) -> ResolutionPlan:
    if path is None:
        return ResolutionPlan(
            aliases={},
            drop_node_ids=set(),
            source_path=None,
            source_format="none",
            approved_same_entity=0,
            applied_aliases=0,
            stale_decisions=0,
            ignored_decisions=0,
            rejected_decisions=0,
            cluster_count=0,
        )

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Resolution decision file not found: {resolved}"
        )

    if resolved.suffix.lower() == ".jsonl":
        return _load_jsonl_resolution_plan(resolved, graph=graph)
    if resolved.suffix.lower() == ".json":
        return _load_legacy_resolution_json(resolved)
    raise ValueError(
        "Resolution decision file must end in .jsonl (Milestone 2A) "
        "or .json (legacy aliases)."
    )


def load_resolution_decisions(
    path: str | Path | None,
    *,
    graph: nx.Graph | None = None,
) -> tuple[dict[str, str], set[str]]:
    """Backward-compatible tuple API.

    JSONL decisions require the graph so node types and stale decisions can be
    validated. Legacy JSON aliases remain loadable without a graph.
    """
    if path is None:
        return {}, set()
    resolved = Path(path)
    if resolved.suffix.lower() == ".jsonl" and graph is None:
        raise ValueError("graph is required when loading JSONL decisions.")
    if graph is None:
        plan = _load_legacy_resolution_json(resolved)
    else:
        plan = load_resolution_plan(resolved, graph=graph)
    return plan.aliases, plan.drop_node_ids


def resolve_alias(
    node_id: str,
    aliases: Mapping[str, str],
) -> str:
    seen: set[str] = set()
    current = node_id

    while current in aliases:
        if current in seen:
            cycle = " -> ".join([*seen, current])
            raise ValueError(f"Alias cycle detected: {cycle}")
        seen.add(current)
        current = aliases[current]

    return current


def merge_node_attributes(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    existing_type = str(existing.get("type", ""))
    incoming_type = str(incoming.get("type", ""))

    if existing_type and incoming_type and existing_type != incoming_type:
        raise ValueError(
            "Cannot merge nodes with different types: "
            f"{existing_type!r} vs {incoming_type!r}"
        )

    merged = dict(existing)
    conflicts: dict[str, list[Any]] = {}

    for key, value in incoming.items():
        if key not in merged or merged[key] in {"", None}:
            merged[key] = value
            continue

        if value in {"", None} or merged[key] == value:
            continue

        if key in {"label", "description", "method_details"}:
            conflicts.setdefault(key, [merged[key]])
            if value not in conflicts[key]:
                conflicts[key].append(value)

    if conflicts:
        existing_conflicts = merged.get("attribute_conflicts_json")
        if existing_conflicts:
            try:
                prior = json.loads(str(existing_conflicts))
            except json.JSONDecodeError:
                prior = {}
        else:
            prior = {}

        for key, values in conflicts.items():
            prior_values = prior.setdefault(key, [])
            for value in values:
                if value not in prior_values:
                    prior_values.append(value)

        merged["attribute_conflicts_json"] = json.dumps(
            prior,
            ensure_ascii=False,
            sort_keys=True,
        )

    return merged


def canonicalize_paper_graph(
    graph: nx.Graph,
    *,
    aliases: Mapping[str, str] | None = None,
    drop_node_ids: set[str] | None = None,
) -> nx.MultiDiGraph:
    """Apply externally reviewed paper-level decisions.

    No candidate is merged merely because it was generated. Only aliases built
    from approved same_entity decisions (or explicitly supplied legacy aliases)
    should be passed here.
    """
    aliases = dict(aliases or {})
    drop_node_ids = set(drop_node_ids or set())

    canonical = nx.MultiDiGraph()
    canonical.graph.update(graph.graph)

    aliases_by_canonical: dict[str, set[str]] = defaultdict(set)
    canonical_id_map = {
        str(node_id): resolve_alias(str(node_id), aliases)
        for node_id in graph.nodes
        if str(node_id) not in drop_node_ids
    }

    for node_id, node_data in graph.nodes(data=True):
        node_id = str(node_id)
        if node_id in drop_node_ids:
            continue

        canonical_id = canonical_id_map[node_id]
        aliases_by_canonical[canonical_id].add(node_id)
        remapped_node_data = remap_node_reference_attributes(
            dict(node_data),
            canonical_id_map,
        )

        if canonical_id in canonical:
            merged_data = merge_node_attributes(
                dict(canonical.nodes[canonical_id]),
                remapped_node_data,
            )
            canonical.nodes[canonical_id].update(merged_data)
        else:
            canonical.add_node(canonical_id, **remapped_node_data)

    for canonical_id, member_ids in aliases_by_canonical.items():
        canonical.nodes[canonical_id]["aliases_json"] = json.dumps(
            sorted(member_ids),
            ensure_ascii=False,
        )
        canonical.nodes[canonical_id]["resolution_member_count"] = len(
            member_ids
        )

    if graph.is_multigraph():
        edge_iterator = graph.edges(keys=True, data=True)
    else:
        edge_iterator = (
            (source, target, str(index), edge_data)
            for index, (source, target, edge_data)
            in enumerate(graph.edges(data=True))
        )

    for index, (source, target, original_key, edge_data) in enumerate(
        edge_iterator
    ):
        source = str(source)
        target = str(target)

        if source in drop_node_ids or target in drop_node_ids:
            continue

        canonical_source = resolve_alias(source, aliases)
        canonical_target = resolve_alias(target, aliases)

        if canonical_source == canonical_target:
            continue

        chunk_id = str(edge_data.get("chunk_id", "unknown_chunk"))
        edge_key = f"{chunk_id}:{original_key}:{index}"

        canonical.add_edge(
            canonical_source,
            canonical_target,
            key=edge_key,
            **dict(edge_data),
        )

    return canonical
