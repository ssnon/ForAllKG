from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


@dataclass(frozen=True)
class CorpusVisualizationBundle:
    corpus_dir: Path
    graph: nx.MultiDiGraph
    manifest: dict[str, Any]
    audit: dict[str, Any]
    node_rows: list[dict[str, Any]]
    evidence_rows: list[dict[str, Any]]
    registry_rows: list[dict[str, Any]]
    pattern_rows: list[dict[str, Any]]
    candidate_rows: list[dict[str, Any]]


_NODE_FILL = {
    "Catalyst": "#4C78A8",
    "CatalystModel": "#72A0C1",
    "Experiment": "#F58518",
    "Calculation": "#E45756",
    "Measurement": "#54A24B",
    "MeasurementGroup": "#8CD17D",
    "ObservationClaim": "#B279A2",
    "MechanismClaim": "#7A5195",
    "BridgeConcept": "#ECA82C",
    "Metal": "#9D755D",
    "Support": "#BAB0AC",
    "CoordinationMotif": "#FF9DA6",
    "Reaction": "#59A14F",
    "ReactionStep": "#76B7B2",
    "Intermediate": "#EDC948",
    "Material": "#A0CBE8",
    "SynthesisMethod": "#FFBE7D",
    "Precursor": "#D4A6C8",
    "CorpusAlignment": "#3A86FF",
    "CorpusPattern": "#8338EC",
    "Unknown": "#9CA3AF",
}

_PAPER_COLORS = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#393B79",
    "#637939",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected object in {path}:{line_number}."
                )
            rows.append(value)
    return rows


def load_corpus_visualization_bundle(
    corpus_dir: str | Path,
) -> CorpusVisualizationBundle:
    corpus_dir = Path(corpus_dir)
    graph_path = corpus_dir / "graph.graphml"
    manifest_path = corpus_dir / "manifest.json"
    audit_path = corpus_dir / "audit.json"

    required = (graph_path, manifest_path, audit_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Corpus bundle is incomplete:\n- " + "\n- ".join(missing)
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(audit, dict):
        raise ValueError("Invalid corpus manifest or audit payload.")

    graph = nx.read_graphml(graph_path, force_multigraph=True)
    if int(manifest.get("corpus_nodes", -1)) != graph.number_of_nodes():
        raise ValueError("Corpus node count does not match manifest.")
    if int(manifest.get("corpus_edges", -1)) != graph.number_of_edges():
        raise ValueError("Corpus edge count does not match manifest.")

    return CorpusVisualizationBundle(
        corpus_dir=corpus_dir,
        graph=graph,
        manifest=manifest,
        audit=audit,
        node_rows=_read_jsonl(corpus_dir / "node_text.jsonl"),
        evidence_rows=_read_jsonl(corpus_dir / "edge_evidence.jsonl"),
        registry_rows=_read_jsonl(
            corpus_dir / "registry_alignments.jsonl"
        ),
        pattern_rows=_read_jsonl(
            corpus_dir / "pattern_alignments.jsonl"
        ),
        candidate_rows=_read_jsonl(
            corpus_dir / "cross_paper_resolution_candidates.jsonl"
        ),
    )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return list(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _node_label(node_id: str, attrs: dict[str, Any]) -> str:
    return str(
        attrs.get("label")
        or attrs.get("statement")
        or attrs.get("name")
        or attrs.get("metric")
        or node_id
    )


def _natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in parts
    )


def _paper_color_map(paper_ids: Iterable[str]) -> dict[str, str]:
    ordered = sorted(set(map(str, paper_ids)), key=_natural_key)
    return {
        paper_id: _PAPER_COLORS[index % len(_PAPER_COLORS)]
        for index, paper_id in enumerate(ordered)
    }


def _shape_for_node(node_type: str, kind: str, attrs: dict[str, Any]) -> str:
    if kind == "alignment_hub":
        if node_type == "CorpusPattern":
            return "hexagon"
        return "diamond"
    if node_type in {"ObservationClaim", "MechanismClaim"}:
        return "round-rectangle"
    if node_type == "BridgeConcept":
        return "round-rectangle"
    if node_type in {"Experiment", "Calculation"}:
        return "rectangle"
    if node_type in {"Measurement", "MeasurementGroup"}:
        return "ellipse"
    return "ellipse"


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# ---------------------------------------------------------------------------
# Viewer data
# ---------------------------------------------------------------------------


def _evidence_index(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        edge_id = str(row.get("projection_edge_id", "")).strip()
        if edge_id:
            index[edge_id].append(dict(row))
    return index


def _edge_ids(attrs: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    edge_id = str(attrs.get("edge_id", "")).strip()
    if edge_id:
        values.add(edge_id)
    for field in (
        "projection_edge_ids_json",
        "source_edge_ids_json",
    ):
        values.update(
            str(item)
            for item in _json_list(attrs.get(field))
            if str(item).strip()
        )
    return values


def _compact_evidence(
    rows: Iterable[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        signature = json.dumps(
            {
                "paper": row.get("source_paper_id")
                or row.get("source_paper_ids_json"),
                "relation": row.get("relation"),
                "status": row.get("evidence_status"),
                "text": row.get("evidence_text"),
                "rule": row.get("derivation_rule"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if signature in seen:
            continue
        seen.add(signature)
        compact.append({
            "sourcePaperId": str(
                row.get("source_paper_id", "") or ""
            ),
            "sourcePaperIds": [
                str(item)
                for item in _json_list(row.get("source_paper_ids_json"))
            ],
            "relation": str(row.get("relation", "")),
            "evidenceStatus": str(row.get("evidence_status", "")),
            "graphLayer": str(row.get("graph_layer", "")),
            "evidenceText": _truncate(row.get("evidence_text", ""), 700),
            "evidencePointers": _json_list(
                row.get("evidence_pointers_json")
            ),
            "derivationRule": str(row.get("derivation_rule", "")),
            "requiresVerification": _as_bool(
                row.get("requires_verification", False)
            ),
        })
        if len(compact) >= limit:
            break
    return compact


def build_viewer_payload(
    bundle: CorpusVisualizationBundle,
    *,
    max_evidence_per_edge: int = 4,
) -> dict[str, Any]:
    graph = bundle.graph
    paper_ids = [str(item) for item in bundle.manifest.get("paper_ids", [])]
    if not paper_ids:
        paper_ids = sorted({
            str(attrs.get("source_paper_id", ""))
            for _, attrs in graph.nodes(data=True)
            if str(attrs.get("source_paper_id", "")).strip()
        }, key=_natural_key)
    paper_colors = _paper_color_map(paper_ids)
    evidence_by_edge = _evidence_index(bundle.evidence_rows)

    nodes: list[dict[str, Any]] = []
    node_type_counts: Counter[str] = Counter()
    per_paper_node_counts: Counter[str] = Counter()

    for node_id, raw_attrs in graph.nodes(data=True):
        attrs = dict(raw_attrs)
        node_id = str(node_id)
        node_type = str(attrs.get("type", "Unknown")) or "Unknown"
        kind = str(attrs.get("corpus_node_kind", "paper_local"))
        paper_id = str(attrs.get("source_paper_id", ""))
        paper_list = [
            str(item)
            for item in _json_list(attrs.get("source_paper_ids_json"))
            if str(item).strip()
        ]
        if paper_id and paper_id not in paper_list:
            paper_list.append(paper_id)
        paper_list = sorted(set(paper_list), key=_natural_key)
        retention_lane = str(attrs.get("retention_lane", ""))
        requires_verification = _as_bool(
            attrs.get("requires_verification", False)
        ) or retention_lane == "semantic_candidate"
        visual_group = (
            "CorpusPattern"
            if node_type == "CorpusPattern"
            else "CorpusAlignment"
            if node_type == "CorpusAlignment"
            else node_type
        )

        node_type_counts[node_type] += 1
        if paper_id:
            per_paper_node_counts[paper_id] += 1

        nodes.append({
            "id": node_id,
            "label": _truncate(_node_label(node_id, attrs), 120),
            "fullLabel": _node_label(node_id, attrs),
            "type": node_type,
            "visualGroup": visual_group,
            "kind": kind,
            "paperId": paper_id,
            "paperIds": paper_list,
            "paperColor": paper_colors.get(paper_id, "#6B7280"),
            "fillColor": _NODE_FILL.get(node_type, _NODE_FILL["Unknown"]),
            "shape": _shape_for_node(node_type, kind, attrs),
            "graphLayer": str(attrs.get("graph_layer", "")),
            "evidenceStatus": str(attrs.get("evidence_status", "")),
            "retentionLane": retention_lane,
            "conceptType": str(attrs.get("concept_type", "")),
            "alignmentType": str(attrs.get("alignment_type", "")),
            "entityType": str(attrs.get("entity_type", "")),
            "patternSubject": str(attrs.get("pattern_subject", "")),
            "patternRelation": str(attrs.get("pattern_relation", "")),
            "patternObject": str(attrs.get("pattern_object", "")),
            "supportCount": max(1, _as_int(attrs.get("support_count"), 1)),
            "requiresVerification": requires_verification,
            "nodeText": _truncate(attrs.get("node_text", ""), 3000),
            "sourceNodeId": str(attrs.get("source_node_id", "")),
        })

    # Collapse parallel corpus edges for readability while preserving their
    # relation/evidence sets in the details panel.
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    per_paper_edge_counts: Counter[str] = Counter()

    if graph.is_multigraph():
        iterator = graph.edges(keys=True, data=True)
    else:
        iterator = (
            (left, right, str(index), attrs)
            for index, (left, right, attrs)
            in enumerate(graph.edges(data=True))
        )

    for left, right, key, raw_attrs in iterator:
        left = str(left)
        right = str(right)
        attrs = dict(raw_attrs)
        signature = (left, right)
        row = grouped.setdefault(signature, {
            "id": f"ui::{len(grouped)}::{left}::{right}",
            "source": left,
            "target": right,
            "relations": set(),
            "evidenceStatuses": set(),
            "graphLayers": set(),
            "sourcePaperIds": set(),
            "edgeIds": set(),
            "supportCount": 0,
            "requiresVerificationValues": [],
            "explorationCost": math.inf,
            "alignment": False,
            "evidence": [],
            "rawEdgeCount": 0,
        })
        relation = str(attrs.get("relation", "RELATED_TO"))
        row["relations"].add(relation)
        status = str(attrs.get("evidence_status", ""))
        if status:
            row["evidenceStatuses"].add(status)
        layer = str(attrs.get("graph_layer", ""))
        if layer:
            row["graphLayers"].add(layer)
        row["sourcePaperIds"].update(
            str(item)
            for item in _json_list(attrs.get("source_paper_ids_json"))
            if str(item).strip()
        )
        source_paper = str(attrs.get("source_paper_id", ""))
        if source_paper:
            row["sourcePaperIds"].add(source_paper)
            per_paper_edge_counts[source_paper] += 1
        ids = _edge_ids(attrs)
        row["edgeIds"].update(ids)
        row["supportCount"] += max(1, _as_int(attrs.get("support_count"), 1))
        row["requiresVerificationValues"].append(
            _as_bool(attrs.get("requires_verification", False))
        )
        row["explorationCost"] = min(
            row["explorationCost"],
            _as_float(attrs.get("exploration_cost"), 1.0),
        )
        row["alignment"] = row["alignment"] or (
            str(attrs.get("corpus_edge_kind", "")) == "alignment"
        )
        row["rawEdgeCount"] += 1

        evidence_rows: list[dict[str, Any]] = []
        for edge_id in ids:
            evidence_rows.extend(evidence_by_edge.get(edge_id, []))
        row["evidence"].extend(evidence_rows)

    edges: list[dict[str, Any]] = []
    for row in grouped.values():
        statuses = sorted(row["evidenceStatuses"])
        layers = sorted(row["graphLayers"])
        relations = sorted(row["relations"])
        has_unverified = any(row["requiresVerificationValues"])
        all_unverified = bool(row["requiresVerificationValues"]) and all(
            row["requiresVerificationValues"]
        )
        derived = any(
            "derived" in status.lower()
            for status in statuses
        ) or any("derived" in layer.lower() for layer in layers)
        line_style = "dotted" if all_unverified else "dashed" if derived else "solid"
        edges.append({
            "id": row["id"],
            "source": row["source"],
            "target": row["target"],
            "relation": relations[0] if len(relations) == 1 else "MULTI_RELATION",
            "relations": relations,
            "title": " / ".join(relations),
            "evidenceStatuses": statuses,
            "graphLayers": layers,
            "sourcePaperIds": sorted(row["sourcePaperIds"], key=_natural_key),
            "edgeIds": sorted(row["edgeIds"]),
            "supportCount": max(1, int(row["supportCount"])),
            "rawEdgeCount": int(row["rawEdgeCount"]),
            "requiresVerification": all_unverified,
            "hasVerificationRequired": has_unverified,
            "explorationCost": (
                1.0 if math.isinf(row["explorationCost"])
                else float(row["explorationCost"])
            ),
            "alignment": bool(row["alignment"]),
            "lineStyle": line_style,
            "evidence": _compact_evidence(
                row["evidence"],
                limit=max_evidence_per_edge,
            ),
        })

    hub_rows = [
        node
        for node in nodes
        if node["kind"] == "alignment_hub"
    ]
    hub_by_id = {node["id"]: node for node in hub_rows}

    paper_summaries: list[dict[str, Any]] = []
    for paper_id in sorted(paper_ids, key=_natural_key):
        hubs = [
            hub
            for hub in hub_rows
            if paper_id in hub["paperIds"]
        ]
        paper_summaries.append({
            "paperId": paper_id,
            "color": paper_colors.get(paper_id, "#6B7280"),
            "nodes": per_paper_node_counts[paper_id],
            "sourceEdges": per_paper_edge_counts[paper_id],
            "alignmentHubs": len(hubs),
            "registryHubs": sum(
                hub["type"] == "CorpusAlignment" for hub in hubs
            ),
            "patternHubs": sum(
                hub["type"] == "CorpusPattern" for hub in hubs
            ),
        })

    similarity = build_paper_similarity(
        paper_ids=paper_ids,
        hubs=hub_rows,
    )

    top_hubs = sorted(
        hub_rows,
        key=lambda row: (
            -len(row["paperIds"]),
            -row["supportCount"],
            row["label"],
        ),
    )

    return {
        "meta": {
            "corpusId": str(bundle.manifest.get("corpus_id", "")),
            "mode": str(bundle.manifest.get("mode", "")),
            "paperCount": len(paper_ids),
            "paperIds": sorted(paper_ids, key=_natural_key),
            "nodes": len(nodes),
            "collapsedEdges": len(edges),
            "rawEdges": graph.number_of_edges(),
            "alignmentHubs": len(hub_rows),
            "registryAlignmentHubs": sum(
                node["type"] == "CorpusAlignment" for node in hub_rows
            ),
            "patternAlignmentHubs": sum(
                node["type"] == "CorpusPattern" for node in hub_rows
            ),
            "reviewCandidates": len(bundle.candidate_rows),
            "passesStructuralGate": bool(
                bundle.audit.get("passes_structural_gate", False)
            ),
        },
        "papers": paper_summaries,
        "nodeTypeCounts": dict(sorted(node_type_counts.items())),
        "nodes": nodes,
        "edges": edges,
        "hubs": top_hubs,
        "hubById": {
            hub_id: {
                "paperIds": hub["paperIds"],
                "label": hub["label"],
                "type": hub["type"],
                "supportCount": hub["supportCount"],
            }
            for hub_id, hub in hub_by_id.items()
        },
        "paperSimilarity": similarity,
    }


# ---------------------------------------------------------------------------
# Paper-to-paper summary
# ---------------------------------------------------------------------------


def build_paper_similarity(
    *,
    paper_ids: Iterable[str],
    hubs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    papers = sorted(set(map(str, paper_ids)), key=_natural_key)
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"registry": 0, "pattern": 0}
    )
    for hub in hubs:
        members = sorted(set(map(str, hub.get("paperIds", []))), key=_natural_key)
        kind = "pattern" if hub.get("type") == "CorpusPattern" else "registry"
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                key = tuple(sorted((left, right), key=_natural_key))
                counts[key][kind] += 1

    rows: list[dict[str, Any]] = []
    for left in papers:
        for right in papers:
            if left == right:
                registry = pattern = 0
            else:
                key = tuple(sorted((left, right), key=_natural_key))
                registry = counts[key]["registry"]
                pattern = counts[key]["pattern"]
            rows.append({
                "paperA": left,
                "paperB": right,
                "registryShared": registry,
                "patternShared": pattern,
                "totalShared": registry + pattern,
            })
    return rows


def write_paper_similarity_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = (
        "paperA",
        "paperB",
        "registryShared",
        "patternShared",
        "totalShared",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Static SVG overview
# ---------------------------------------------------------------------------


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "middle",
    size: int = 15,
    weight: int = 400,
    fill: str = "#111827",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'
    )


def render_overview_svg(
    payload: dict[str, Any],
    *,
    max_hubs: int = 30,
    width: int = 1600,
    height: int = 1050,
) -> str:
    papers = list(payload.get("papers", []))
    hubs = list(payload.get("hubs", []))[: max(1, max_hubs)]
    cx = width / 2
    cy = height / 2 + 20
    outer_radius = min(width, height) * 0.39

    paper_positions: dict[str, tuple[float, float]] = {}
    for index, paper in enumerate(papers):
        angle = -math.pi / 2 + 2 * math.pi * index / max(1, len(papers))
        paper_positions[paper["paperId"]] = (
            cx + outer_radius * math.cos(angle),
            cy + outer_radius * math.sin(angle),
        )

    hub_positions: dict[str, tuple[float, float]] = {}
    golden = math.pi * (3 - math.sqrt(5))
    for index, hub in enumerate(hubs):
        radial_fraction = math.sqrt((index + 1) / max(1, len(hubs)))
        radius = outer_radius * 0.58 * radial_fraction
        angle = index * golden
        hub_positions[hub["id"]] = (
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
        )

    lines: list[str] = []
    for hub in hubs:
        hx, hy = hub_positions[hub["id"]]
        for paper_id in hub.get("paperIds", []):
            if paper_id not in paper_positions:
                continue
            px, py = paper_positions[paper_id]
            lines.append(
                f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" '
                'stroke="#CBD5E1" stroke-width="1.4" opacity="0.52" />'
            )

    hub_shapes: list[str] = []
    for index, hub in enumerate(hubs):
        hx, hy = hub_positions[hub["id"]]
        paper_count = max(1, len(hub.get("paperIds", [])))
        radius = 8 + min(20, 2.4 * paper_count)
        fill = "#8338EC" if hub.get("type") == "CorpusPattern" else "#3A86FF"
        if hub.get("type") == "CorpusPattern":
            points = []
            for j in range(6):
                angle = math.pi / 6 + j * math.pi / 3
                points.append(
                    f"{hx + radius * math.cos(angle):.1f},"
                    f"{hy + radius * math.sin(angle):.1f}"
                )
            hub_shapes.append(
                f'<polygon points="{" ".join(points)}" fill="{fill}" '
                'stroke="#FFFFFF" stroke-width="2" opacity="0.92" />'
            )
        else:
            hub_shapes.append(
                f'<rect x="{hx-radius:.1f}" y="{hy-radius:.1f}" '
                f'width="{2*radius:.1f}" height="{2*radius:.1f}" '
                f'transform="rotate(45 {hx:.1f} {hy:.1f})" fill="{fill}" '
                'stroke="#FFFFFF" stroke-width="2" opacity="0.92" />'
            )
        if index < min(16, len(hubs)):
            label = _truncate(hub.get("label", ""), 34)
            hub_shapes.append(
                _svg_text(
                    hx,
                    hy + radius + 17,
                    label,
                    size=11,
                    fill="#374151",
                )
            )

    paper_shapes: list[str] = []
    for paper in papers:
        px, py = paper_positions[paper["paperId"]]
        paper_shapes.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="31" '
            f'fill="{paper["color"]}" stroke="#FFFFFF" stroke-width="4" />'
        )
        paper_shapes.append(
            _svg_text(
                px,
                py + 5,
                paper["paperId"],
                size=12,
                weight=700,
                fill="#FFFFFF",
            )
        )

    title = _svg_text(
        width / 2,
        42,
        f'{payload.get("meta", {}).get("corpusId", "Corpus")} · cross-paper alignment overview',
        size=24,
        weight=700,
    )
    subtitle = _svg_text(
        width / 2,
        69,
        "Outer nodes are papers; inner hubs are shared registry entities and confirmed Bridge patterns.",
        size=13,
        fill="#4B5563",
    )

    legend = [
        '<g transform="translate(30,88)">',
        '<rect x="0" y="0" width="320" height="92" rx="10" fill="#FFFFFF" stroke="#E5E7EB" />',
        '<rect x="18" y="21" width="14" height="14" transform="rotate(45 25 28)" fill="#3A86FF" />',
        _svg_text(44, 33, "Registry alignment hub", anchor="start", size=12),
        '<polygon points="25,49 32,53 32,61 25,65 18,61 18,53" fill="#8338EC" />',
        _svg_text(44, 61, "Confirmed cross-paper pattern hub", anchor="start", size=12),
        _svg_text(18, 82, f"Showing top {len(hubs)} hubs by paper/support coverage", anchor="start", size=11, fill="#6B7280"),
        '</g>',
    ]

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Cross-paper corpus alignment overview</title>',
        '<desc id="desc">Papers around the outside are connected to shared registry and confirmed pattern hubs.</desc>',
        '<rect width="100%" height="100%" fill="#FFFFFF" />',
        title,
        subtitle,
        *legend,
        '<g>',
        *lines,
        '</g>',
        '<g>',
        *hub_shapes,
        '</g>',
        '<g>',
        *paper_shapes,
        '</g>',
        '</svg>',
    ])


def render_similarity_svg(
    payload: dict[str, Any],
    *,
    width: int = 1050,
    height: int = 980,
) -> str:
    papers = [paper["paperId"] for paper in payload.get("papers", [])]
    rows = payload.get("paperSimilarity", [])
    values = {
        (str(row["paperA"]), str(row["paperB"])): int(row["totalShared"])
        for row in rows
    }
    max_value = max(values.values(), default=0)
    left = 165
    top = 125
    available = min(width - left - 60, height - top - 80)
    cell = available / max(1, len(papers))

    cells: list[str] = []
    labels: list[str] = []
    for i, paper_a in enumerate(papers):
        y = top + i * cell
        labels.append(
            _svg_text(
                left - 12,
                y + cell * 0.62,
                paper_a,
                anchor="end",
                size=12,
            )
        )
        labels.append(
            f'<g transform="translate({left + i * cell + cell * 0.55:.1f},{top - 12:.1f}) rotate(-45)">'
            + _svg_text(0, 0, paper_a, anchor="start", size=12)
            + '</g>'
        )
        for j, paper_b in enumerate(papers):
            x = left + j * cell
            value = values.get((paper_a, paper_b), 0)
            if paper_a == paper_b:
                fill = "#F3F4F6"
                text_fill = "#9CA3AF"
            else:
                ratio = 0 if max_value <= 0 else value / max_value
                # Light-to-dark blue, generated without external dependencies.
                r = round(232 - 156 * ratio)
                g = round(241 - 110 * ratio)
                b = round(250 - 54 * ratio)
                fill = f"rgb({r},{g},{b})"
                text_fill = "#111827" if ratio < 0.55 else "#FFFFFF"
            cells.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'fill="{fill}" stroke="#FFFFFF" stroke-width="1" />'
            )
            if paper_a != paper_b:
                cells.append(
                    _svg_text(
                        x + cell / 2,
                        y + cell * 0.62,
                        str(value),
                        size=11,
                        weight=600,
                        fill=text_fill,
                    )
                )

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Paper similarity matrix</title>',
        '<desc id="desc">Each cell counts registry and confirmed pattern hubs shared by a pair of papers.</desc>',
        '<rect width="100%" height="100%" fill="#FFFFFF" />',
        _svg_text(width / 2, 40, "Cross-paper shared-hub matrix", size=24, weight=700),
        _svg_text(width / 2, 67, "Cell value = shared registry hubs + shared confirmed Bridge pattern hubs", size=13, fill="#4B5563"),
        *cells,
        *labels,
        '</svg>',
    ])


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------


_VIEWER_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.31.0/dist/cytoscape.min.js"></script>
<style>
:root {
  color-scheme: light dark;
  --bg: #f7f8fb;
  --panel: #ffffff;
  --border: #d9dee8;
  --text: #172033;
  --muted: #697386;
  --accent: #2563eb;
  --chip: #eef2ff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111827;
    --panel: #182132;
    --border: #334155;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #60a5fa;
    --chip: #23314e;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }
body { min-height: 100vh; }
.app { display: grid; grid-template-rows: auto 1fr; min-height: 100vh; }
.topbar { padding: 12px 16px; border-bottom: 1px solid var(--border); background: var(--panel); display: grid; gap: 10px; }
.titleline { display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline; justify-content: space-between; }
.titleline h1 { margin: 0; font-size: 18px; font-weight: 700; }
.metrics { display: flex; flex-wrap: wrap; gap: 8px; }
.metric { padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px; font-size: 12px; background: var(--chip); }
.controls { display: grid; grid-template-columns: minmax(160px, 1.3fr) repeat(3, minmax(120px, .7fr)) auto auto; gap: 8px; align-items: end; }
label { font-size: 11px; color: var(--muted); display: grid; gap: 4px; }
input, select, button { font: inherit; }
input, select { width: 100%; border: 1px solid var(--border); border-radius: 8px; background: var(--panel); color: var(--text); padding: 8px 9px; }
button { border: 1px solid var(--border); border-radius: 8px; background: var(--panel); color: var(--text); padding: 8px 10px; cursor: pointer; }
button.primary { background: var(--accent); color: white; border-color: var(--accent); }
.main { min-height: 0; display: grid; grid-template-columns: 230px minmax(0, 1fr) 310px; }
.sidebar, .details { min-height: 0; overflow: auto; background: var(--panel); padding: 12px; }
.sidebar { border-right: 1px solid var(--border); }
.details { border-left: 1px solid var(--border); }
#cy { min-height: 620px; height: calc(100vh - 132px); background: var(--bg); }
.section { margin-bottom: 16px; }
.section h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 8px; }
.checkrow { display: flex; align-items: center; gap: 7px; margin: 6px 0; font-size: 12px; }
.checkrow input { width: auto; }
.paper-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }
.legend-row { display: grid; grid-template-columns: 14px 1fr; gap: 8px; align-items: center; font-size: 12px; margin: 6px 0; }
.legend-shape { width: 12px; height: 12px; border-radius: 50%; border: 2px solid transparent; }
.legend-line { height: 0; border-top: 2px solid currentColor; }
.legend-line.dashed { border-top-style: dashed; }
.legend-line.dotted { border-top-style: dotted; }
.detail-title { font-size: 16px; font-weight: 700; margin: 0 0 6px; overflow-wrap: anywhere; }
.badges { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
.badge { font-size: 11px; border: 1px solid var(--border); border-radius: 999px; padding: 3px 7px; background: var(--chip); }
.kv { display: grid; grid-template-columns: 98px 1fr; gap: 6px 8px; font-size: 12px; margin: 8px 0 12px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; overflow-wrap: anywhere; }
.textbox { border: 1px solid var(--border); border-radius: 8px; padding: 8px; font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; background: var(--bg); margin-bottom: 8px; }
.pathbar { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px; }
.pathslot { border: 1px solid var(--border); border-radius: 8px; padding: 7px; min-height: 45px; font-size: 11px; overflow-wrap: anywhere; }
.pathslot strong { display: block; color: var(--muted); margin-bottom: 3px; }
.path-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.notice { font-size: 11px; color: var(--muted); line-height: 1.45; }
.evidence-item { border-top: 1px solid var(--border); padding-top: 8px; margin-top: 8px; }
.evidence-item:first-child { border-top: 0; padding-top: 0; }
.path-summary { margin-top: 10px; padding: 8px; border: 1px solid var(--border); border-radius: 8px; font-size: 12px; }
@media (max-width: 980px) {
  .controls { grid-template-columns: 1fr 1fr; }
  .main { grid-template-columns: 190px minmax(0, 1fr); }
  .details { grid-column: 1 / -1; border-left: 0; border-top: 1px solid var(--border); max-height: 360px; }
  #cy { height: 620px; }
}
@media (max-width: 680px) {
  .main { grid-template-columns: 1fr; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--border); max-height: 280px; }
  .controls { grid-template-columns: 1fr; }
  #cy { height: 560px; }
}
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="titleline">
      <h1 id="title"></h1>
      <div class="metrics" id="metrics"></div>
    </div>
    <div class="controls">
      <label>Search
        <input id="search" type="search" placeholder="NiFe, HER, overpotential, ..." />
      </label>
      <label>View
        <select id="viewMode">
          <option value="cross">Cross-paper context</option>
          <option value="full">Full corpus</option>
          <option value="hubs">Alignment hubs</option>
        </select>
      </label>
      <label>Layout
        <select id="layoutMode">
          <option value="cose">Force-directed</option>
          <option value="concentric">Concentric</option>
          <option value="breadthfirst">Breadth-first</option>
          <option value="circle">Circle</option>
        </select>
      </label>
      <label>Neighborhood
        <select id="neighborDepth">
          <option value="0">Match only</option>
          <option value="1" selected>1 hop</option>
          <option value="2">2 hops</option>
        </select>
      </label>
      <button id="runLayout" class="primary" type="button">Run layout</button>
      <button id="reset" type="button">Reset</button>
    </div>
  </div>
  <div class="main">
    <aside class="sidebar">
      <div class="section">
        <h2>Papers</h2>
        <div id="paperFilters"></div>
      </div>
      <div class="section">
        <h2>Node types</h2>
        <div id="typeFilters"></div>
      </div>
      <div class="section">
        <h2>Evidence</h2>
        <label class="checkrow"><input type="checkbox" id="showVerified" checked /> Verified / asserted</label>
        <label class="checkrow"><input type="checkbox" id="showDerived" checked /> Derived projection</label>
        <label class="checkrow"><input type="checkbox" id="showCandidates" checked /> Verification-required candidates</label>
      </div>
      <div class="section">
        <h2>Encoding</h2>
        <div class="legend-row"><span class="legend-shape" style="background:#3A86FF"></span><span>Registry alignment hub</span></div>
        <div class="legend-row"><span class="legend-shape" style="background:#8338EC"></span><span>Confirmed pattern hub</span></div>
        <div class="legend-row"><span class="legend-line"></span><span>Source-asserted</span></div>
        <div class="legend-row"><span class="legend-line dashed"></span><span>Derived</span></div>
        <div class="legend-row"><span class="legend-line dotted"></span><span>Verification required</span></div>
      </div>
      <div class="notice">Default view keeps cross-paper hubs, their mentions, and one local context hop. Use “Full corpus” when you need every extracted node and edge.</div>
    </aside>
    <main id="cy" aria-label="Interactive knowledge graph"></main>
    <aside class="details">
      <div class="section">
        <h2>Selected item</h2>
        <div id="detailsBody" class="notice">Click a node or edge.</div>
      </div>
      <div class="section">
        <h2>Reasoning path</h2>
        <div class="pathbar">
          <div class="pathslot"><strong>Start</strong><span id="pathStart">Not set</span></div>
          <div class="pathslot"><strong>Target</strong><span id="pathTarget">Not set</span></div>
        </div>
        <div class="path-actions">
          <button id="findPath" type="button">Shortest path</button>
          <button id="nearestCross" class="primary" type="button">Nearest cross-paper</button>
          <button id="clearPath" type="button">Clear path</button>
          <button id="fit" type="button">Fit graph</button>
        </div>
        <div id="pathSummary" class="path-summary notice">Select a node, then use the detail panel to set it as the start or target.</div>
      </div>
    </aside>
  </div>
</div>
<script>
window.CORPUS_DATA = __DATA__;
</script>
<script>
(function () {
  const data = window.CORPUS_DATA;
  const meta = data.meta;
  document.getElementById('title').textContent = `${meta.corpusId} · ${meta.mode} knowledge graph`;
  document.getElementById('metrics').innerHTML = [
    `${meta.paperCount} papers`,
    `${meta.nodes.toLocaleString()} nodes`,
    `${meta.rawEdges.toLocaleString()} raw edges`,
    `${meta.alignmentHubs} cross-paper hubs`,
    `structural gate: ${meta.passesStructuralGate ? 'PASS' : 'FAIL'}`
  ].map(value => `<span class="metric">${escapeHtml(value)}</span>`).join('');

  if (typeof cytoscape === 'undefined') {
    document.getElementById('cy').innerHTML = '<div style="padding:24px">Cytoscape.js could not be loaded. Internet access to jsDelivr is required for the interactive viewer. The static overview SVG files remain available offline.</div>';
    return;
  }

  const nodeById = new Map(data.nodes.map(n => [n.id, n]));
  const edgeById = new Map(data.edges.map(e => [e.id, e]));
  const paperSet = new Set(data.papers.map(p => p.paperId));

  const elements = [
    ...data.nodes.map(n => ({ data: { ...n, labelDisplay: shorten(n.label, 36) } })),
    ...data.edges.map(e => ({ data: { ...e, widthValue: Math.min(8, 1 + Math.log2(1 + e.supportCount)) } }))
  ];

  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements,
    minZoom: 0.05,
    maxZoom: 3.5,
    wheelSensitivity: 0.22,
    style: [
      {
        selector: 'node',
        style: {
          'background-color': 'data(fillColor)',
          'border-color': 'data(paperColor)',
          'border-width': 3,
          'shape': 'data(shape)',
          'label': 'data(labelDisplay)',
          'font-size': 9,
          'text-wrap': 'wrap',
          'text-max-width': 90,
          'text-valign': 'bottom',
          'text-margin-y': 5,
          'color': '#374151',
          'width': ele => Math.min(54, 20 + 4 * Math.log2(1 + Number(ele.data('supportCount') || 1))),
          'height': ele => Math.min(54, 20 + 4 * Math.log2(1 + Number(ele.data('supportCount') || 1))),
          'opacity': 0.92
        }
      },
      {
        selector: 'node[kind = "alignment_hub"]',
        style: {
          'border-color': '#ffffff',
          'border-width': 2,
          'font-weight': 700,
          'font-size': 10,
          'width': ele => Math.min(72, 32 + 6 * Math.log2(1 + Number(ele.data('supportCount') || 1))),
          'height': ele => Math.min(72, 32 + 6 * Math.log2(1 + Number(ele.data('supportCount') || 1)))
        }
      },
      {
        selector: 'node[requiresVerification = true]',
        style: {
          'border-style': 'dashed',
          'opacity': 0.62
        }
      },
      {
        selector: 'edge',
        style: {
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'arrow-scale': 0.65,
          'line-color': '#94A3B8',
          'target-arrow-color': '#94A3B8',
          'width': 'data(widthValue)',
          'opacity': 0.48,
          'line-style': 'data(lineStyle)'
        }
      },
      {
        selector: 'edge[alignment = true]',
        style: {
          'line-color': '#64748B',
          'target-arrow-color': '#64748B',
          'opacity': 0.34
        }
      },
      {
        selector: '.faded',
        style: { 'opacity': 0.07 }
      },
      {
        selector: '.highlighted',
        style: {
          'opacity': 1,
          'z-index': 999,
          'border-width': 5,
          'line-color': '#ef4444',
          'target-arrow-color': '#ef4444'
        }
      },
      {
        selector: '.search-hit',
        style: {
          'opacity': 1,
          'border-width': 5,
          'border-color': '#f59e0b',
          'z-index': 900
        }
      }
    ],
    layout: { name: 'cose', animate: false, fit: true, padding: 45, nodeRepulsion: 180000, idealEdgeLength: 85 }
  });

  function shorten(value, limit) {
    const text = String(value || '');
    return text.length <= limit ? text : text.slice(0, limit - 1) + '…';
  }
  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>'"]/g, ch => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;' }[ch]));
  }
  function unique(values) { return [...new Set(values.filter(Boolean))]; }
  function formatList(values) { return unique(values || []).join(', ') || '—'; }
  function nodePapers(node) { return unique([node.paperId, ...(node.paperIds || [])]); }

  const paperFilters = document.getElementById('paperFilters');
  data.papers.forEach(paper => {
    const id = `paper-${paper.paperId.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
    const row = document.createElement('label');
    row.className = 'checkrow';
    row.innerHTML = `<input type="checkbox" data-paper="${escapeHtml(paper.paperId)}" id="${id}" checked />` +
      `<span class="paper-dot" style="background:${paper.color}"></span>` +
      `<span>${escapeHtml(paper.paperId)} (${paper.nodes})</span>`;
    paperFilters.appendChild(row);
  });

  const typeFilters = document.getElementById('typeFilters');
  Object.entries(data.nodeTypeCounts).sort((a,b) => b[1]-a[1]).forEach(([type, count]) => {
    const id = `type-${type.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
    const row = document.createElement('label');
    row.className = 'checkrow';
    const fill = (data.nodes.find(n => n.type === type) || {}).fillColor || '#9CA3AF';
    row.innerHTML = `<input type="checkbox" data-type="${escapeHtml(type)}" id="${id}" checked />` +
      `<span class="legend-shape" style="background:${fill}"></span>` +
      `<span>${escapeHtml(type)} (${count})</span>`;
    typeFilters.appendChild(row);
  });

  let pathStartId = null;
  let pathTargetId = null;

  function selectedPapers() {
    return new Set([...document.querySelectorAll('[data-paper]')].filter(el => el.checked).map(el => el.dataset.paper));
  }
  function selectedTypes() {
    return new Set([...document.querySelectorAll('[data-type]')].filter(el => el.checked).map(el => el.dataset.type));
  }
  function evidenceAllowedNode(node) {
    if (node.data('requiresVerification') === true || node.data('requiresVerification') === 'true') {
      return document.getElementById('showCandidates').checked;
    }
    const status = String(node.data('evidenceStatus') || '').toLowerCase();
    if (status.includes('derived')) return document.getElementById('showDerived').checked;
    return document.getElementById('showVerified').checked;
  }
  function evidenceAllowedEdge(edge) {
    if (edge.data('requiresVerification') === true || edge.data('requiresVerification') === 'true') {
      return document.getElementById('showCandidates').checked;
    }
    const statuses = edge.data('evidenceStatuses') || [];
    const derived = statuses.some(s => String(s).toLowerCase().includes('derived'));
    if (derived) return document.getElementById('showDerived').checked;
    return document.getElementById('showVerified').checked;
  }

  function crossPaperContextSet() {
    const result = new Set();
    cy.nodes('[kind = "alignment_hub"]').forEach(hub => {
      result.add(hub.id());
      hub.neighborhood('node').forEach(member => {
        result.add(member.id());
        member.neighborhood('node').forEach(local => result.add(local.id()));
      });
    });
    return result;
  }

  function searchContextSet(query, depth) {
    const q = query.trim().toLowerCase();
    if (!q) return null;
    const result = new Set();
    cy.nodes().forEach(node => {
      const haystack = [node.data('fullLabel'), node.data('type'), node.data('nodeText'), node.data('paperId')].join(' ').toLowerCase();
      if (!haystack.includes(q)) return;
      result.add(node.id());
      let frontier = node.collection();
      for (let i = 0; i < depth; i++) {
        frontier = frontier.neighborhood('node');
        frontier.forEach(n => result.add(n.id()));
      }
    });
    return result;
  }

  function applyFilters(runLayoutAfter = false) {
    const papers = selectedPapers();
    const types = selectedTypes();
    const view = document.getElementById('viewMode').value;
    const depth = Number(document.getElementById('neighborDepth').value || 1);
    const query = document.getElementById('search').value;
    const searchSet = searchContextSet(query, depth);
    const crossSet = view === 'cross' ? crossPaperContextSet() : null;

    cy.batch(() => {
      cy.elements().removeClass('search-hit');
      cy.nodes().forEach(node => {
        const paperIds = nodePapers(node.data());
        const paperAllowed = !paperIds.length || paperIds.some(p => papers.has(p));
        const typeAllowed = types.has(node.data('type'));
        const viewAllowed = view === 'full' ||
          (view === 'hubs' && (node.data('kind') === 'alignment_hub' || node.neighborhood('node[kind = "alignment_hub"]').length > 0)) ||
          (view === 'cross' && crossSet && crossSet.has(node.id()));
        const searchAllowed = !searchSet || searchSet.has(node.id());
        const evidenceAllowed = evidenceAllowedNode(node);
        if (paperAllowed && typeAllowed && viewAllowed && searchAllowed && evidenceAllowed) node.show();
        else node.hide();
        if (searchSet && searchSet.has(node.id()) && [node.data('fullLabel'), node.data('type'), node.data('nodeText')].join(' ').toLowerCase().includes(query.trim().toLowerCase())) {
          node.addClass('search-hit');
        }
      });
      cy.edges().forEach(edge => {
        const endpointsVisible = edge.source().visible() && edge.target().visible();
        if (endpointsVisible && evidenceAllowedEdge(edge)) edge.show();
        else edge.hide();
      });
    });
    if (runLayoutAfter) runLayout();
  }

  function runLayout() {
    const name = document.getElementById('layoutMode').value;
    const visible = cy.elements(':visible');
    let options = { name, animate: false, fit: true, padding: 35 };
    if (name === 'cose') options = { ...options, nodeRepulsion: 180000, idealEdgeLength: 85, gravity: 0.22, componentSpacing: 80 };
    if (name === 'concentric') options = { ...options, concentric: n => n.data('kind') === 'alignment_hub' ? 3 : 1, levelWidth: () => 1 };
    visible.layout(options).run();
  }

  function resetGraph() {
    document.getElementById('search').value = '';
    document.getElementById('viewMode').value = 'cross';
    [...document.querySelectorAll('[data-paper], [data-type]')].forEach(el => el.checked = true);
    document.getElementById('showVerified').checked = true;
    document.getElementById('showDerived').checked = true;
    document.getElementById('showCandidates').checked = true;
    clearPath();
    applyFilters(true);
  }

  function setStart(id) {
    pathStartId = id;
    document.getElementById('pathStart').textContent = nodeById.get(id)?.fullLabel || id;
  }
  function setTarget(id) {
    pathTargetId = id;
    document.getElementById('pathTarget').textContent = nodeById.get(id)?.fullLabel || id;
  }

  function clearPath() {
    pathStartId = null;
    pathTargetId = null;
    document.getElementById('pathStart').textContent = 'Not set';
    document.getElementById('pathTarget').textContent = 'Not set';
    document.getElementById('pathSummary').textContent = 'Select a node, then use the detail panel to set it as the start or target.';
    cy.elements().removeClass('highlighted faded');
  }

  function highlightPath(path) {
    cy.elements().removeClass('highlighted faded');
    cy.elements().addClass('faded');
    path.removeClass('faded').addClass('highlighted').show();
    path.connectedEdges().removeClass('faded');
    cy.fit(path, 60);
    const papers = unique(path.nodes().flatMap(n => nodePapers(n.data()))).filter(p => paperSet.has(p));
    const labels = path.nodes().map(n => n.data('fullLabel') || n.id());
    document.getElementById('pathSummary').innerHTML = `<strong>${path.nodes().length} nodes · ${path.edges().length} edges</strong><br>` +
      `papers: ${escapeHtml(papers.join(', ') || '—')}<br>` +
      escapeHtml(labels.join(' → '));
  }

  function shortestPath() {
    if (!pathStartId || !pathTargetId) {
      document.getElementById('pathSummary').textContent = 'Set both start and target nodes first.';
      return;
    }
    const dijkstra = cy.elements().dijkstra({
      root: cy.$id(pathStartId),
      directed: false,
      weight: edge => Number(edge.data('explorationCost') || 1)
    });
    const path = dijkstra.pathTo(cy.$id(pathTargetId));
    if (!path || !path.length) {
      document.getElementById('pathSummary').textContent = 'No path found.';
      return;
    }
    highlightPath(path);
  }

  function nearestCrossPaperPath() {
    if (!pathStartId) {
      document.getElementById('pathSummary').textContent = 'Set a start node first.';
      return;
    }
    const start = cy.$id(pathStartId);
    if (!start.length) return;
    const startPapers = new Set(nodePapers(start.data()).filter(p => paperSet.has(p)));
    if (!startPapers.size) {
      document.getElementById('pathSummary').textContent = 'The selected start node is a corpus hub. Choose a paper-local node for nearest cross-paper search.';
      return;
    }

    const queue = [pathStartId];
    const visited = new Set([pathStartId]);
    const prevNode = new Map();
    const prevEdge = new Map();
    let found = null;

    while (queue.length && !found) {
      const currentId = queue.shift();
      const current = cy.$id(currentId);
      current.connectedEdges().forEach(edge => {
        if (found) return;
        const next = edge.source().id() === currentId ? edge.target() : edge.source();
        const nextId = next.id();
        if (visited.has(nextId)) return;
        visited.add(nextId);
        prevNode.set(nextId, currentId);
        prevEdge.set(nextId, edge.id());
        const nextPapers = nodePapers(next.data()).filter(p => paperSet.has(p));
        if (nextPapers.some(p => !startPapers.has(p))) {
          found = nextId;
          return;
        }
        queue.push(nextId);
      });
    }

    if (!found) {
      document.getElementById('pathSummary').textContent = 'No cross-paper path found.';
      return;
    }

    const ids = [];
    const edgeIds = [];
    let cursor = found;
    ids.push(cursor);
    while (cursor !== pathStartId) {
      edgeIds.push(prevEdge.get(cursor));
      cursor = prevNode.get(cursor);
      ids.push(cursor);
    }
    ids.reverse();
    edgeIds.reverse();
    let path = cy.collection();
    ids.forEach(id => { path = path.union(cy.$id(id)); });
    edgeIds.forEach(id => { path = path.union(cy.$id(id)); });
    pathTargetId = found;
    document.getElementById('pathTarget').textContent = nodeById.get(found)?.fullLabel || found;
    highlightPath(path);
  }

  function nodeDetails(node) {
    const d = node.data();
    const body = document.getElementById('detailsBody');
    body.innerHTML = `
      <h3 class="detail-title">${escapeHtml(d.fullLabel || d.id)}</h3>
      <div class="badges">
        <span class="badge">${escapeHtml(d.type)}</span>
        ${d.paperId ? `<span class="badge">${escapeHtml(d.paperId)}</span>` : ''}
        ${d.kind === 'alignment_hub' ? '<span class="badge">cross-paper hub</span>' : ''}
        ${d.requiresVerification ? '<span class="badge">verification required</span>' : ''}
      </div>
      <dl class="kv">
        <dt>ID</dt><dd>${escapeHtml(d.id)}</dd>
        <dt>Papers</dt><dd>${escapeHtml(formatList(nodePapers(d)))}</dd>
        <dt>Layer</dt><dd>${escapeHtml(d.graphLayer || '—')}</dd>
        <dt>Status</dt><dd>${escapeHtml(d.evidenceStatus || d.retentionLane || '—')}</dd>
        <dt>Support</dt><dd>${escapeHtml(d.supportCount)}</dd>
        ${d.patternRelation ? `<dt>Pattern</dt><dd>${escapeHtml(`${d.patternSubject} ${d.patternRelation} ${d.patternObject}`)}</dd>` : ''}
      </dl>
      ${d.nodeText ? `<div class="textbox">${escapeHtml(d.nodeText)}</div>` : ''}
      <div class="path-actions">
        <button id="detailStart" type="button">Set as start</button>
        <button id="detailTarget" type="button">Set as target</button>
      </div>`;
    document.getElementById('detailStart').addEventListener('click', () => setStart(d.id));
    document.getElementById('detailTarget').addEventListener('click', () => setTarget(d.id));
  }

  function edgeDetails(edge) {
    const d = edge.data();
    const body = document.getElementById('detailsBody');
    const source = nodeById.get(d.source);
    const target = nodeById.get(d.target);
    const evidence = (d.evidence || []).map(row => `
      <div class="evidence-item">
        <div class="badges">
          ${row.sourcePaperId ? `<span class="badge">${escapeHtml(row.sourcePaperId)}</span>` : ''}
          <span class="badge">${escapeHtml(row.evidenceStatus || 'evidence')}</span>
        </div>
        ${row.evidenceText ? `<div class="textbox">${escapeHtml(row.evidenceText)}</div>` : ''}
        ${row.derivationRule ? `<div class="notice">derivation: ${escapeHtml(row.derivationRule)}</div>` : ''}
      </div>`).join('');
    body.innerHTML = `
      <h3 class="detail-title">${escapeHtml(d.title || d.relation)}</h3>
      <div class="badges">
        ${d.alignment ? '<span class="badge">corpus alignment</span>' : ''}
        ${d.requiresVerification ? '<span class="badge">verification required</span>' : ''}
        <span class="badge">support ${escapeHtml(d.supportCount)}</span>
      </div>
      <dl class="kv">
        <dt>Source</dt><dd>${escapeHtml(source?.fullLabel || d.source)}</dd>
        <dt>Target</dt><dd>${escapeHtml(target?.fullLabel || d.target)}</dd>
        <dt>Relations</dt><dd>${escapeHtml(formatList(d.relations))}</dd>
        <dt>Status</dt><dd>${escapeHtml(formatList(d.evidenceStatuses))}</dd>
        <dt>Papers</dt><dd>${escapeHtml(formatList(d.sourcePaperIds))}</dd>
        <dt>Cost</dt><dd>${escapeHtml(d.explorationCost)}</dd>
      </dl>
      ${evidence || '<div class="notice">No compact evidence excerpt was embedded for this edge. Inspect edge_evidence.jsonl for the full sidecar.</div>'}`;
  }

  cy.on('tap', 'node', evt => nodeDetails(evt.target));
  cy.on('tap', 'edge', evt => edgeDetails(evt.target));
  cy.on('tap', evt => {
    if (evt.target === cy) document.getElementById('detailsBody').textContent = 'Click a node or edge.';
  });

  document.getElementById('search').addEventListener('input', () => applyFilters(false));
  document.getElementById('viewMode').addEventListener('change', () => applyFilters(true));
  document.getElementById('neighborDepth').addEventListener('change', () => applyFilters(false));
  [...document.querySelectorAll('[data-paper], [data-type], #showVerified, #showDerived, #showCandidates')].forEach(el => el.addEventListener('change', () => applyFilters(false)));
  document.getElementById('runLayout').addEventListener('click', runLayout);
  document.getElementById('reset').addEventListener('click', resetGraph);
  document.getElementById('findPath').addEventListener('click', shortestPath);
  document.getElementById('nearestCross').addEventListener('click', nearestCrossPaperPath);
  document.getElementById('clearPath').addEventListener('click', clearPath);
  document.getElementById('fit').addEventListener('click', () => cy.fit(cy.elements(':visible'), 40));

  applyFilters(true);
})();
</script>
</body>
</html>
'''


def render_viewer_html(payload: dict[str, Any]) -> str:
    title = (
        f'{payload.get("meta", {}).get("corpusId", "Corpus")} '
        f'{payload.get("meta", {}).get("mode", "")} graph viewer'
    ).strip()
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return (
        _VIEWER_TEMPLATE
        .replace("__TITLE__", html.escape(title))
        .replace("__DATA__", encoded)
    )


# ---------------------------------------------------------------------------
# End-to-end output
# ---------------------------------------------------------------------------


def build_corpus_visualization(
    bundle: CorpusVisualizationBundle,
    *,
    output_dir: str | Path,
    max_overview_hubs: int = 30,
    max_evidence_per_edge: int = 4,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_viewer_payload(
        bundle,
        max_evidence_per_edge=max_evidence_per_edge,
    )

    data_path = output_dir / "visualization_data.json"
    data_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    viewer_path = output_dir / "viewer.html"
    viewer_path.write_text(
        render_viewer_html(payload),
        encoding="utf-8",
    )

    overview_path = output_dir / "overview.svg"
    overview_path.write_text(
        render_overview_svg(
            payload,
            max_hubs=max_overview_hubs,
        ),
        encoding="utf-8",
    )

    similarity_svg_path = output_dir / "paper_similarity.svg"
    similarity_svg_path.write_text(
        render_similarity_svg(payload),
        encoding="utf-8",
    )

    similarity_csv_path = write_paper_similarity_csv(
        output_dir / "paper_similarity.csv",
        payload["paperSimilarity"],
    )

    summary = {
        "corpus_id": payload["meta"]["corpusId"],
        "mode": payload["meta"]["mode"],
        "papers": payload["meta"]["paperCount"],
        "nodes": payload["meta"]["nodes"],
        "raw_edges": payload["meta"]["rawEdges"],
        "collapsed_viewer_edges": payload["meta"]["collapsedEdges"],
        "alignment_hubs": payload["meta"]["alignmentHubs"],
        "registry_alignment_hubs": payload["meta"]["registryAlignmentHubs"],
        "pattern_alignment_hubs": payload["meta"]["patternAlignmentHubs"],
        "review_candidates": payload["meta"]["reviewCandidates"],
        "source_structural_gate": payload["meta"]["passesStructuralGate"],
        "max_overview_hubs": max_overview_hubs,
        "max_evidence_per_edge": max_evidence_per_edge,
        "viewer": str(viewer_path),
        "overview_svg": str(overview_path),
        "paper_similarity_svg": str(similarity_svg_path),
        "paper_similarity_csv": str(similarity_csv_path),
        "visualization_data": str(data_path),
        "cytoscape_runtime": (
            "https://cdn.jsdelivr.net/npm/cytoscape@3.31.0/"
            "dist/cytoscape.min.js"
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
