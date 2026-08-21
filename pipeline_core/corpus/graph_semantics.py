from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from pipeline_core.corpus.graph.graph_domain import GraphDomainAdapter

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


MappingLike = dict[str, Any]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _json(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _safe_add_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    key: str,
    attrs: dict[str, Any],
) -> None:
    final_key = key
    suffix = 1
    while graph.has_edge(source, target, final_key):
        final_key = f"{key}:domain_canonicalization:{suffix}"
        suffix += 1
    graph.add_edge(source, target, key=final_key, **attrs)


def _merge_text_attr(current: str | None, incoming: str | None) -> str:
    current = current or ""
    incoming = incoming or ""
    if not current:
        return incoming
    if not incoming:
        return current
    return current if len(current) >= len(incoming) else incoming


def _merge_alias_json(existing: Any, *extra_ids: str) -> str:
    aliases: set[str] = set()
    if existing:
        try:
            payload = json.loads(str(existing))
            if isinstance(payload, list):
                aliases.update(str(x) for x in payload)
        except Exception:
            aliases.add(str(existing))
    aliases.update(extra_ids)
    return json.dumps(sorted(aliases), ensure_ascii=False)


def _paper_node_score(
    node_id: str,
    attrs: MappingLike,
    *,
    paper_id: str,
) -> tuple[int, int, str]:
    label = str(attrs.get("label", ""))
    description = str(attrs.get("description", ""))
    normalized_label = label.strip().lower()
    normalized_paper = paper_id.strip().lower()
    informative = int(bool(label) and normalized_label != normalized_paper)
    length = len(label) + len(description)
    return (informative, length, str(node_id))


def merge_same_paper_nodes(
    graph: nx.MultiDiGraph,
    *,
    paper_id: str,
) -> tuple[nx.MultiDiGraph, list[dict[str, Any]]]:
    paper_nodes = [
        (str(node_id), dict(attrs))
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("type", "")) == "Paper"
    ]
    if len(paper_nodes) <= 1:
        return graph, []

    canonical_id, canonical_attrs = max(
        paper_nodes,
        key=lambda row: _paper_node_score(row[0], row[1], paper_id=paper_id),
    )
    rows: list[dict[str, Any]] = []

    for alias_id, alias_attrs in sorted(paper_nodes):
        if alias_id == canonical_id:
            continue

        graph.nodes[canonical_id]["label"] = _merge_text_attr(
            str(graph.nodes[canonical_id].get("label", "")),
            str(alias_attrs.get("label", "")),
        )
        graph.nodes[canonical_id]["description"] = _merge_text_attr(
            str(graph.nodes[canonical_id].get("description", "")),
            str(alias_attrs.get("description", "")),
        )
        graph.nodes[canonical_id]["aliases_json"] = _merge_alias_json(
            graph.nodes[canonical_id].get("aliases_json"),
            canonical_id,
            alias_id,
        )
        graph.nodes[canonical_id]["paper_identity_canonicalization"] = "same_paper_id"

        for source, _, key, edge_attrs in list(graph.in_edges(alias_id, keys=True, data=True)):
            source_id = canonical_id if source == alias_id else str(source)
            _safe_add_edge(
                graph,
                source_id,
                canonical_id,
                str(key),
                dict(edge_attrs),
            )

        for _, target, key, edge_attrs in list(graph.out_edges(alias_id, keys=True, data=True)):
            target_id = canonical_id if target == alias_id else str(target)
            _safe_add_edge(
                graph,
                canonical_id,
                target_id,
                str(key),
                dict(edge_attrs),
            )

        graph.remove_node(alias_id)
        rows.append({
            "action": "merge_same_paper_id",
            "paper_id": paper_id,
            "canonical_node_id": canonical_id,
            "alias_node_id": alias_id,
            "alias_label": alias_attrs.get("label", ""),
        })

    return graph, rows


def apply_graph_domain_canonicalization(
    graph: nx.MultiDiGraph,
    *,
    graph_adapter: GraphDomainAdapter,
    paper_id: str,
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    graph, paper_rows = merge_same_paper_nodes(graph, paper_id=paper_id)
    rows.extend(paper_rows)

    summary = {
        "adapter_id": graph_adapter.adapter_id,
        "paper_identity_merges": len(paper_rows),
        "actions": rows,
    }
    return graph, summary


def duplicate_label_groups(
    graph: nx.MultiDiGraph,
    *,
    review_node_types: frozenset[str],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for node_id, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("type", ""))
        label = str(attrs.get("label", "")).strip().lower()
        if not label:
            continue
        buckets[(node_type, label)].append(str(node_id))

    rows: list[dict[str, Any]] = []
    for (node_type, label), node_ids in sorted(buckets.items()):
        if len(node_ids) < 2:
            continue
        rows.append({
            "node_type": node_type,
            "normalized_label": label,
            "count": len(node_ids),
            "node_ids": node_ids,
            "severity": (
                "review"
                if node_type in review_node_types
                else "info"
            ),
        })
    return rows


def component_diagnostics(
    graph: nx.MultiDiGraph,
    *,
    primary_subject_types: frozenset[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, component in enumerate(nx.weakly_connected_components(graph), start=1):
        node_ids = sorted(str(node_id) for node_id in component)
        types = sorted({
            str(graph.nodes[node_id].get("type", ""))
            for node_id in component
        })
        has_paper = "Paper" in types
        has_primary = bool(
            primary_subject_types & set(types)
        )
        rows.append({
            "component_index": index,
            "node_count": len(node_ids),
            "edge_count": graph.subgraph(component).number_of_edges(),
            "node_types": types,
            "contains_paper": has_paper,
            "contains_primary_subject": has_primary,
            "severity": "warning" if not has_paper and not has_primary else "info",
            "sample_node_ids": node_ids[:12],
        })
    return rows


def write_graph_semantics_report(
    run_dir: Path,
    graph: nx.MultiDiGraph,
    *,
    graph_adapter: GraphDomainAdapter,
) -> dict[str, Any]:
    output_dir = run_dir / "graph_semantics"
    output_dir.mkdir(parents=True, exist_ok=True)

    relation_issues = [
        issue.to_dict()
        for issue in graph_adapter.diagnose_relation_contracts(graph)
    ]
    domain_diagnostics = graph_adapter.collect_diagnostics(graph)

    diagnostics_version = str(
        domain_diagnostics.get(
            "diagnostics_version",
            "graph-domain-diagnostics-v1",
        )
    )
    role_issues = list(
        domain_diagnostics.get("node_role_issues", [])
    )
    evidence_topology_issues = list(
        domain_diagnostics.get("evidence_topology_issues", [])
    )
    relation_triage = list(
        domain_diagnostics.get("relation_triage", [])
    )
    relation_direction_issues = list(
        domain_diagnostics.get("relation_direction_issues", [])
    )
    integration_components = list(
        domain_diagnostics.get("integration_components", [])
    )
    component_bridge_candidates = list(
        domain_diagnostics.get("component_bridge_candidates", [])
    )

    duplicates = duplicate_label_groups(
        graph,
        review_node_types=graph_adapter.duplicate_review_types,
    )
    components = component_diagnostics(
        graph,
        primary_subject_types=graph_adapter.primary_subject_types,
    )

    _write_json(
        output_dir / "relation_contract_issues.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "issue_count": len(relation_issues),
            "issues": relation_issues,
        },
    )
    _write_csv(output_dir / "relation_contract_issues.csv", relation_issues)

    _write_json(
        output_dir / "node_role_issues.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "issue_count": len(role_issues),
            "issues": role_issues,
        },
    )
    _write_csv(output_dir / "node_role_issues.csv", role_issues)

    _write_json(
        output_dir / "evidence_topology_issues.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "issue_count": len(evidence_topology_issues),
            "issues": evidence_topology_issues,
        },
    )
    _write_csv(output_dir / "evidence_topology_issues.csv", evidence_topology_issues)

    _write_json(
        output_dir / "relation_contract_triage.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "count": len(relation_triage),
            "rows": relation_triage,
        },
    )
    _write_csv(
        output_dir / "relation_contract_triage.csv",
        relation_triage,
    )

    _write_json(
        output_dir / "relation_direction_issues.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "count": len(relation_direction_issues),
            "rows": relation_direction_issues,
        },
    )
    _write_csv(
        output_dir / "relation_direction_issues.csv",
        relation_direction_issues,
    )

    _write_json(
        output_dir / "integration_components.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "count": len(integration_components),
            "components": integration_components,
        },
    )
    _write_csv(
        output_dir / "integration_components.csv",
        integration_components,
    )

    _write_json(
        output_dir / "component_bridge_candidates.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "count": len(component_bridge_candidates),
            "candidates": component_bridge_candidates,
        },
    )
    _write_csv(
        output_dir / "component_bridge_candidates.csv",
        component_bridge_candidates,
    )

    _write_json(
        output_dir / "duplicate_label_groups.json",
        {
            "count": len(duplicates),
            "groups": duplicates,
        },
    )
    _write_csv(output_dir / "duplicate_label_groups.csv", duplicates)

    _write_json(
        output_dir / "components.json",
        {
            "count": len(components),
            "components": components,
        },
    )
    _write_csv(output_dir / "components.csv", components)

    warning_count = sum(
        1
        for row in relation_issues
        if str(row.get("severity", "")).lower() == "warning"
    )
    error_count = sum(
        1
        for row in relation_issues
        if str(row.get("severity", "")).lower() == "error"
    )
    summary = {
        "adapter_id": graph_adapter.adapter_id,
        "diagnostics_version": diagnostics_version,
        "relation_contract_issue_count": len(relation_issues),
        "relation_contract_warning_count": warning_count,
        "relation_contract_error_count": error_count,
        "node_role_issue_count": len(role_issues),
        "evidence_topology_issue_count": len(evidence_topology_issues),
        "relation_triage_count": len(relation_triage),
        "relation_direction_issue_count": len(relation_direction_issues),
        "relation_triage_category_counts": dict(sorted({
            category: sum(
                1
                for row in relation_triage
                if row["category"] == category
            )
            for category in {
                row["category"]
                for row in relation_triage
            }
        }.items())),
        "integration_review_component_count": sum(
            1
            for row in integration_components
            if row["severity"] == "review"
        ),
        "integration_component_subtype_counts": dict(sorted({
            subtype: sum(
                1
                for row in integration_components
                if row.get("component_subtype") == subtype
            )
            for subtype in {
                row.get("component_subtype", "")
                for row in integration_components
                if row.get("component_subtype")
            }
        }.items())),
        "component_bridge_candidate_count": len(component_bridge_candidates),
        "duplicate_label_group_count": len(duplicates),
        "component_count": len(components),
        "non_primary_component_count": sum(
            1
            for row in components
            if not row["contains_paper"] and not row["contains_primary_subject"]
        ),
        "report_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
