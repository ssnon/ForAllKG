from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.domains.catalysis_mechanism_graph import (
    BROAD_DIRECT_MECHANISM_RELATIONS,
    BROAD_MECHANISM_CORE_TYPES,
)


@dataclass(frozen=True)
class MechanismSignature:
    source_type: str
    relation: str
    target_type: str
    paper_ids: tuple[str, ...]
    edge_count: int

    @property
    def paper_support(self) -> int:
        return len(self.paper_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "relation": self.relation,
            "target_type": self.target_type,
            "paper_ids": list(self.paper_ids),
            "paper_support": self.paper_support,
            "edge_count": self.edge_count,
        }


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _edge_papers(attrs: dict[str, Any]) -> set[str]:
    papers = {
        str(value)
        for value in _json_list(attrs.get("source_paper_ids_json"))
        if str(value).strip()
    }
    direct = str(attrs.get("source_paper_id") or "").strip()
    if direct:
        papers.add(direct)
    return papers


def mechanism_signatures(
    graph: nx.Graph,
) -> list[MechanismSignature]:
    grouped_papers: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    grouped_edges: Counter[tuple[str, str, str]] = Counter()

    for source, target, attrs in graph.edges(data=True):
        if str(attrs.get("graph_layer", "")) == "corpus_alignment":
            continue
        relation = str(attrs.get("relation", ""))
        if relation not in BROAD_DIRECT_MECHANISM_RELATIONS:
            continue
        source_type = str(graph.nodes[source].get("type", ""))
        target_type = str(graph.nodes[target].get("type", ""))
        signature = (source_type, relation, target_type)
        grouped_edges[signature] += 1
        grouped_papers[signature].update(_edge_papers(dict(attrs)))

    rows = [
        MechanismSignature(
            source_type=source_type,
            relation=relation,
            target_type=target_type,
            paper_ids=tuple(sorted(grouped_papers[signature])),
            edge_count=grouped_edges[signature],
        )
        for signature in sorted(grouped_edges)
        for source_type, relation, target_type in [signature]
    ]
    return sorted(
        rows,
        key=lambda row: (
            -row.paper_support,
            -row.edge_count,
            row.relation,
            row.source_type,
            row.target_type,
        ),
    )


def audit_broad_corpus(
    graph: nx.Graph,
    *,
    expected_paper_ids: Iterable[str] = (),
) -> tuple[dict[str, Any], list[MechanismSignature]]:
    expected = [str(value) for value in expected_paper_ids]
    per_paper_nodes: Counter[str] = Counter()
    per_paper_edges: Counter[str] = Counter()
    per_paper_mechanism_edges: Counter[str] = Counter()
    node_type_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()

    for _, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("type", ""))
        node_type_counts[node_type] += 1
        paper_id = str(attrs.get("source_paper_id") or "").strip()
        if paper_id:
            per_paper_nodes[paper_id] += 1

    source_scientific_edges = 0
    direct_mechanism_edges = 0
    verification_required_edges = 0
    for _, _, attrs in graph.edges(data=True):
        graph_layer = str(attrs.get("graph_layer", ""))
        if graph_layer == "corpus_alignment":
            continue
        source_scientific_edges += 1
        relation = str(attrs.get("relation", ""))
        relation_counts[relation] += 1
        if relation in BROAD_DIRECT_MECHANISM_RELATIONS:
            direct_mechanism_edges += 1
        if str(attrs.get("requires_verification", "")).lower() in {
            "1", "true", "yes"
        } or attrs.get("requires_verification") is True:
            verification_required_edges += 1
        papers = _edge_papers(dict(attrs))
        for paper_id in papers:
            per_paper_edges[paper_id] += 1
            if relation in BROAD_DIRECT_MECHANISM_RELATIONS:
                per_paper_mechanism_edges[paper_id] += 1

    observed_papers = sorted(set(per_paper_nodes) | set(per_paper_edges))
    paper_scope = expected or observed_papers
    zero_node_papers = sorted(
        paper_id for paper_id in paper_scope if per_paper_nodes[paper_id] == 0
    )
    zero_edge_papers = sorted(
        paper_id for paper_id in paper_scope if per_paper_edges[paper_id] == 0
    )
    mechanism_bearing_papers = sorted(
        paper_id
        for paper_id in paper_scope
        if per_paper_mechanism_edges[paper_id] > 0
    )

    signatures = mechanism_signatures(graph)
    recurring = [row for row in signatures if row.paper_support >= 2]
    core_node_count = sum(
        count
        for node_type, count in node_type_counts.items()
        if node_type in BROAD_MECHANISM_CORE_TYPES
    )

    report = {
        "schema_version": "graphagentsdac-broad-corpus-audit-v1",
        "expected_paper_count": len(expected),
        "observed_paper_count": len(observed_papers),
        "observed_paper_ids": observed_papers,
        "zero_node_papers": zero_node_papers,
        "zero_edge_papers": zero_edge_papers,
        "mechanism_bearing_paper_count": len(mechanism_bearing_papers),
        "mechanism_bearing_paper_fraction": (
            len(mechanism_bearing_papers) / len(paper_scope)
            if paper_scope
            else 0.0
        ),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "source_scientific_edges": source_scientific_edges,
        "core_mechanism_nodes": core_node_count,
        "direct_mechanism_edges": direct_mechanism_edges,
        "direct_mechanism_edge_fraction": (
            direct_mechanism_edges / source_scientific_edges
            if source_scientific_edges
            else 0.0
        ),
        "verification_required_edge_fraction": (
            verification_required_edges / source_scientific_edges
            if source_scientific_edges
            else 0.0
        ),
        "unique_mechanism_signatures": len(signatures),
        "recurring_mechanism_signatures": len(recurring),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "per_paper": {
            paper_id: {
                "nodes": per_paper_nodes[paper_id],
                "edges": per_paper_edges[paper_id],
                "direct_mechanism_edges": per_paper_mechanism_edges[paper_id],
            }
            for paper_id in paper_scope
        },
    }
    return report, signatures


def write_broad_corpus_audit(
    *,
    graph: nx.Graph,
    output_dir: str | Path,
    expected_paper_ids: Iterable[str] = (),
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report, signatures = audit_broad_corpus(
        graph,
        expected_paper_ids=expected_paper_ids,
    )
    report_path = output_dir / "broad_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    signature_path = output_dir / "mechanism_signatures.jsonl"
    with signature_path.open("w", encoding="utf-8") as handle:
        for row in signatures:
            handle.write(
                json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    return report_path, signature_path
