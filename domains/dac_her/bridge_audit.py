from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from domains.dac_her.bridge_policy import BRIDGE_POLICY_VERSION
from domains.dac_her.scientific_signatures import normalize_scientific_text


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            })


def audit_bridge_graph(
    graph: nx.Graph,
    *,
    rejection_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    concepts: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    anchor_status_counts: Counter[str] = Counter()
    pattern_issues: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    labels: defaultdict[str, list[str]] = defaultdict(list)

    for node_id, attrs_value in graph.nodes(data=True):
        attrs = dict(attrs_value)
        if attrs.get("type") != "BridgeConcept":
            continue
        row = {"node_id": str(node_id), **attrs}
        concepts.append(row)
        labels[normalize_scientific_text(attrs.get("label", ""))].append(str(node_id))

        lane = str(attrs.get("retention_lane", ""))
        if lane == "accepted_pattern":
            missing = [
                key for key in (
                    "pattern_subject",
                    "pattern_relation",
                    "pattern_object",
                    "relation_strength",
                    "pattern_support_mode",
                )
                if not str(attrs.get(key, "")).strip()
            ]
            if missing:
                pattern_issues.append({
                    "node_id": str(node_id),
                    "label": attrs.get("label", ""),
                    "issue": "accepted pattern is missing required fields",
                    "missing_fields": missing,
                })

            relation = str(attrs.get("pattern_relation", ""))
            try:
                qualifier_rows = json.loads(str(attrs.get("qualifiers_json", "[]")))
            except json.JSONDecodeError:
                qualifier_rows = []
            qualifier_map = {
                str(item.get("key", "")).strip().lower(): str(item.get("value", ""))
                for item in qualifier_rows
                if isinstance(item, dict)
            }
            if relation == "COMPETES_WITH" and not qualifier_map.get("competition_target"):
                pattern_issues.append({
                    "node_id": str(node_id),
                    "label": attrs.get("label", ""),
                    "issue": "COMPETES_WITH is missing competition_target qualifier",
                })
            if relation == "COMPETES_FOR" and not qualifier_map.get("competitor_members"):
                pattern_issues.append({
                    "node_id": str(node_id),
                    "label": attrs.get("label", ""),
                    "issue": "COMPETES_FOR is missing competitor_members qualifier",
                })
        elif lane == "paper_local_frontier":
            unexpected = [
                key for key in (
                    "pattern_subject",
                    "pattern_relation",
                    "pattern_object",
                    "relation_strength",
                )
                if str(attrs.get(key, "")).strip()
            ]
            if unexpected:
                pattern_issues.append({
                    "node_id": str(node_id),
                    "label": attrs.get("label", ""),
                    "issue": "frontier concept unexpectedly contains pattern fields",
                    "unexpected_fields": unexpected,
                })
        else:
            pattern_issues.append({
                "node_id": str(node_id),
                "label": attrs.get("label", ""),
                "issue": "unknown retention lane",
                "retention_lane": lane,
            })

    for normalized, node_ids in labels.items():
        if normalized and len(node_ids) > 1:
            duplicate_rows.append({
                "normalized_label": normalized,
                "node_ids": node_ids,
                "count": len(node_ids),
                "issue": "review duplicate bridge mentions; do not auto-merge",
            })

    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
        for _, _, _, attrs in edge_iter:
            relation_counts[str(attrs.get("relation", ""))] += 1
            anchor_status_counts[str(attrs.get("anchor_resolution_status", ""))] += 1
    else:
        for _, _, attrs in graph.edges(data=True):
            relation_counts[str(attrs.get("relation", ""))] += 1
            anchor_status_counts[str(attrs.get("anchor_resolution_status", ""))] += 1

    lanes = Counter(str(row.get("retention_lane", "")) for row in concepts)
    concept_types = Counter(str(row.get("concept_type", "")) for row in concepts)
    evidence_scopes = Counter(str(row.get("evidence_scope", "")) for row in concepts)
    support_modes = Counter(str(row.get("pattern_support_mode", "")) for row in concepts)
    pattern_relations = Counter(str(row.get("pattern_relation", "")) for row in concepts)
    rejections = rejection_rows or []
    rejection_reasons: Counter[str] = Counter()
    for row in rejections:
        reasons = row.get("reason_codes", [])
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except json.JSONDecodeError:
                reasons = [reasons]
        for reason in reasons if isinstance(reasons, list) else []:
            rejection_reasons[str(reason)] += 1

    report = {
        "policy_version": BRIDGE_POLICY_VERSION,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "bridge_concepts": len(concepts),
        "patterns": lanes.get("accepted_pattern", 0),
        "frontier_concepts": lanes.get("paper_local_frontier", 0),
        "retention_lanes": dict(lanes),
        "concept_types": dict(concept_types),
        "evidence_scopes": dict(evidence_scopes),
        "support_modes": dict(support_modes),
        "pattern_relations": dict(pattern_relations),
        "relations": dict(relation_counts),
        "anchor_resolution_statuses": dict(anchor_status_counts),
        "rejected_candidates": len(rejections),
        "rejection_reasons": dict(rejection_reasons),
        "pattern_issues": len(pattern_issues),
        "duplicate_label_groups": len(duplicate_rows),
        "ready_for_projection": (
            len(pattern_issues) == 0
            and anchor_status_counts.get("unresolved_in_canonical", 0) == 0
        ),
        "policy_version": str(
            graph.graph.get(
                "bridge_policy_version",
                BRIDGE_POLICY_VERSION,
            )
        ),
        "prompt_version": str(
            graph.graph.get(
                "bridge_prompt_version",
                "",
            )
        ),
        "bridge_run_id": str(
            graph.graph.get(
                "bridge_run_id",
                "",
            )
        ),
    }
    tables = {
        "concepts": concepts,
        "pattern_issues": pattern_issues,
        "duplicate_candidates": duplicate_rows,
        "rejections": rejections,
    }
    return report, tables


def write_bridge_audit(
    graph: nx.Graph,
    *,
    output_dir: str | Path,
    rejection_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report, tables = audit_bridge_graph(
        graph,
        rejection_rows=rejection_rows,
    )
    (output_dir / "bridge_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "bridge_audit_concepts.csv", tables["concepts"])
    _write_csv(
        output_dir / "bridge_pattern_issues.csv",
        tables["pattern_issues"],
    )
    _write_csv(
        output_dir / "bridge_duplicate_candidates.csv",
        tables["duplicate_candidates"],
    )
    _write_csv(
        output_dir / "bridge_rejections_audit.csv",
        tables["rejections"],
    )
    return report
