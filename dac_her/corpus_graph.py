from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import networkx as nx

from dac_her.node_references import remap_node_reference_attributes
from dac_her.resolution_candidates import (
    AUTO_MERGE_TYPES,
    normalize_scientific_text,
)


ProjectionMode = Literal["evidence", "mechanism", "exploratory"]

# Corpus-level destructive merging is intentionally disabled.  Only exact,
# registry-safe scientific entities receive deterministic alignment hubs.
_REGISTRY_ALIGNMENT_TYPES = frozenset(AUTO_MERGE_TYPES)
_REVIEW_CANDIDATE_TYPES = frozenset({
    "Catalyst",
    "CatalystModel",
    "Support",
    "CoordinationMotif",
    "Material",
    "Intermediate",
})


@dataclass(frozen=True)
class ProjectionBundle:
    paper_id: str
    mode: ProjectionMode
    root: Path
    graph_path: Path
    node_text_path: Path
    edge_evidence_path: Path
    summary_path: Path
    graph: nx.MultiDiGraph
    node_rows: list[dict[str, Any]]
    evidence_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    sha256: dict[str, str]


def _stable_id(*parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{line_number}."
                )
            rows.append(payload)
    return rows


def write_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    return path


def namespace_node_id(paper_id: str, node_id: str) -> str:
    return f"paper::{paper_id}::{node_id}"


def namespace_edge_id(paper_id: str, edge_id: str) -> str:
    return f"paper::{paper_id}::{edge_id}"


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return list(value)
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _remap_json_id_list(
    value: Any,
    id_map: dict[str, str],
) -> str:
    return json.dumps(
        [id_map.get(str(item), str(item)) for item in _json_list(value)],
        ensure_ascii=False,
    )


def _namespace_json_edge_id_list(value: Any, paper_id: str) -> str:
    return json.dumps(
        [namespace_edge_id(paper_id, str(item)) for item in _json_list(value)],
        ensure_ascii=False,
    )


def _paper_ids_json(value: Any, paper_id: str) -> str:
    values = {str(item) for item in _json_list(value) if str(item).strip()}
    values.add(paper_id)
    return json.dumps(sorted(values), ensure_ascii=False)


def load_projection_bundle(
    *,
    project_root: str | Path,
    paper_id: str,
    mode: ProjectionMode = "exploratory",
) -> ProjectionBundle:
    project_root = Path(project_root)
    root = (
        project_root
        / "data_dac"
        / "extracted"
        / paper_id
        / "graphagents"
        / mode
    )
    graph_path = root / "graph.graphml"
    node_text_path = root / "node_text.jsonl"
    edge_evidence_path = root / "edge_evidence.jsonl"
    summary_path = root / "summary.json"

    required = (
        graph_path,
        node_text_path,
        edge_evidence_path,
        summary_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Projection bundle is incomplete:\n- " + "\n- ".join(missing)
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Invalid projection summary: {summary_path}")
    if str(summary.get("paper_id", "")) != paper_id:
        raise ValueError(
            f"Projection summary paper_id mismatch for {paper_id}: "
            f"{summary.get('paper_id')!r}."
        )
    if str(summary.get("mode", "")) != mode:
        raise ValueError(
            f"Projection summary mode mismatch for {paper_id}: "
            f"{summary.get('mode')!r} != {mode!r}."
        )

    graph = nx.read_graphml(graph_path, force_multigraph=True)
    if int(summary.get("nodes", -1)) != graph.number_of_nodes():
        raise ValueError(
            f"Projection node count mismatch for {paper_id}."
        )
    if int(summary.get("edges", -1)) != graph.number_of_edges():
        raise ValueError(
            f"Projection edge count mismatch for {paper_id}."
        )

    return ProjectionBundle(
        paper_id=paper_id,
        mode=mode,
        root=root,
        graph_path=graph_path,
        node_text_path=node_text_path,
        edge_evidence_path=edge_evidence_path,
        summary_path=summary_path,
        graph=graph,
        node_rows=_read_jsonl(node_text_path),
        evidence_rows=_read_jsonl(edge_evidence_path),
        summary=summary,
        sha256={
            "graphml": _sha256_file(graph_path),
            "node_text": _sha256_file(node_text_path),
            "edge_evidence": _sha256_file(edge_evidence_path),
            "summary": _sha256_file(summary_path),
        },
    )


def _remap_node_attrs(
    *,
    paper_id: str,
    node_id: str,
    attrs: dict[str, Any],
    id_map: dict[str, str],
) -> dict[str, Any]:
    remapped = remap_node_reference_attributes(attrs, id_map)
    remapped["source_paper_id"] = paper_id
    remapped["source_node_id"] = node_id
    remapped["source_paper_ids_json"] = _paper_ids_json(
        remapped.get("source_paper_ids_json"),
        paper_id,
    )
    remapped["corpus_node_kind"] = "paper_local"
    return remapped


def _remap_edge_attrs(
    *,
    paper_id: str,
    source: str,
    target: str,
    key: str,
    attrs: dict[str, Any],
    id_map: dict[str, str],
) -> dict[str, Any]:
    remapped = dict(attrs)
    remapped["source_paper_id"] = paper_id
    remapped["source_edge_key"] = key
    remapped["source_edge_source"] = source
    remapped["source_edge_target"] = target
    remapped["source_paper_ids_json"] = _paper_ids_json(
        remapped.get("source_paper_ids_json"),
        paper_id,
    )

    if remapped.get("edge_id"):
        remapped["source_edge_id"] = str(remapped["edge_id"])
        remapped["edge_id"] = namespace_edge_id(
            paper_id,
            str(remapped["edge_id"]),
        )

    for field in (
        "source_edge_ids_json",
        "projection_edge_ids_json",
    ):
        if field in remapped:
            remapped[field] = _namespace_json_edge_id_list(
                remapped[field],
                paper_id,
            )

    if "supporting_node_ids_json" in remapped:
        remapped["supporting_node_ids_json"] = _remap_json_id_list(
            remapped["supporting_node_ids_json"],
            id_map,
        )

    return remapped


def namespace_projection(
    bundle: ProjectionBundle,
) -> tuple[nx.MultiDiGraph, dict[str, str]]:
    source = bundle.graph
    id_map = {
        str(node_id): namespace_node_id(bundle.paper_id, str(node_id))
        for node_id in source.nodes
    }

    result = nx.MultiDiGraph(
        graph_stage="paper_namespaced_projection",
        source_paper_id=bundle.paper_id,
        projection_mode=bundle.mode,
    )

    for raw_node_id, attrs in source.nodes(data=True):
        raw_node_id = str(raw_node_id)
        result.add_node(
            id_map[raw_node_id],
            **_remap_node_attrs(
                paper_id=bundle.paper_id,
                node_id=raw_node_id,
                attrs=dict(attrs),
                id_map=id_map,
            ),
        )

    if source.is_multigraph():
        iterator = source.edges(keys=True, data=True)
    else:
        iterator = (
            (left, right, str(index), attrs)
            for index, (left, right, attrs)
            in enumerate(source.edges(data=True))
        )

    for index, (left, right, key, attrs) in enumerate(iterator):
        left_id = str(left)
        right_id = str(right)
        key_string = str(key)
        result.add_edge(
            id_map[left_id],
            id_map[right_id],
            key=(
                f"paper::{bundle.paper_id}::{key_string}::{index}"
            ),
            **_remap_edge_attrs(
                paper_id=bundle.paper_id,
                source=left_id,
                target=right_id,
                key=key_string,
                attrs=dict(attrs),
                id_map=id_map,
            ),
        )

    return result, id_map


def _node_label(node_id: str, attrs: dict[str, Any]) -> str:
    return str(
        attrs.get("label")
        or attrs.get("statement")
        or attrs.get("name")
        or node_id
    )


def _member_paper_ids(
    graph: nx.Graph,
    members: Iterable[str],
) -> list[str]:
    return sorted({
        str(graph.nodes[member].get("source_paper_id", ""))
        for member in members
        if str(graph.nodes[member].get("source_paper_id", "")).strip()
    })


def _alignment_edge_attrs(
    *,
    relation: str,
    members: list[str],
    paper_ids: list[str],
    derivation_rule: str,
) -> dict[str, Any]:
    return {
        "relation": relation,
        "title": relation,
        "graph_layer": "corpus_alignment",
        "evidence_status": "derived_corpus_alignment",
        "requires_verification": False,
        "source_paper_ids_json": json.dumps(paper_ids, ensure_ascii=False),
        "supporting_node_ids_json": json.dumps(members, ensure_ascii=False),
        "source_edge_ids_json": "[]",
        "projection_edge_ids_json": "[]",
        "evidence_pointers_json": "[]",
        "derivation_rule": derivation_rule,
        "support_count": len(members),
        "corpus_edge_kind": "alignment",
    }


def add_registry_alignment_hubs(
    graph: nx.MultiDiGraph,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for node_id, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("type", ""))
        if node_type not in _REGISTRY_ALIGNMENT_TYPES:
            continue
        normalized = normalize_scientific_text(
            _node_label(str(node_id), dict(attrs))
        )
        if not normalized:
            continue
        grouped.setdefault((node_type, normalized), []).append(str(node_id))

    rows: list[dict[str, Any]] = []
    for (node_type, signature), members in sorted(grouped.items()):
        papers = _member_paper_ids(graph, members)
        if len(papers) < 2:
            continue
        members = sorted(members)
        hub_id = (
            f"corpus::registry::{node_type.lower()}::"
            f"{_stable_id(node_type, signature)}"
        )
        labels = sorted({
            _node_label(member, dict(graph.nodes[member]))
            for member in members
        })
        graph.add_node(
            hub_id,
            type="CorpusAlignment",
            label=(labels[0] if labels else signature),
            alignment_type="registry_entity",
            entity_type=node_type,
            normalized_signature=signature,
            source_member_ids_json=json.dumps(members, ensure_ascii=False),
            source_paper_ids_json=json.dumps(papers, ensure_ascii=False),
            support_count=len(members),
            graph_layer="corpus_alignment",
            evidence_status="derived_corpus_alignment",
            requires_verification=False,
            corpus_node_kind="alignment_hub",
            node_text=(
                f"type: CorpusAlignment\n"
                f"entity type: {node_type}\n"
                f"normalized signature: {signature}\n"
                f"papers: {', '.join(papers)}"
            ),
        )
        edge_attrs = _alignment_edge_attrs(
            relation="ALIGNS_TO_REGISTRY_ENTITY",
            members=members,
            paper_ids=papers,
            derivation_rule="exact_registry_safe_normalized_signature",
        )
        reverse_attrs = _alignment_edge_attrs(
            relation="HAS_PAPER_MENTION",
            members=members,
            paper_ids=papers,
            derivation_rule="exact_registry_safe_normalized_signature",
        )
        for member in members:
            graph.add_edge(
                member,
                hub_id,
                key=f"align:{_stable_id(member, hub_id, 'forward')}",
                **edge_attrs,
            )
            graph.add_edge(
                hub_id,
                member,
                key=f"align:{_stable_id(hub_id, member, 'reverse')}",
                **reverse_attrs,
            )
        rows.append({
            "hub_id": hub_id,
            "alignment_type": "registry_entity",
            "entity_type": node_type,
            "normalized_signature": signature,
            "member_ids": members,
            "paper_ids": papers,
            "member_count": len(members),
        })
    return rows


def _pattern_signature(attrs: dict[str, Any]) -> tuple[str, str, str] | None:
    subject = normalize_scientific_text(attrs.get("pattern_subject", ""))
    relation = normalize_scientific_text(attrs.get("pattern_relation", ""))
    object_value = normalize_scientific_text(attrs.get("pattern_object", ""))
    if not subject or not relation or not object_value:
        return None
    return subject, relation, object_value


def add_pattern_alignment_hubs(
    graph: nx.MultiDiGraph,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for node_id, attrs in graph.nodes(data=True):
        if str(attrs.get("type", "")) != "BridgeConcept":
            continue
        if str(attrs.get("retention_lane", "")) != "accepted_pattern":
            continue
        signature = _pattern_signature(dict(attrs))
        if signature is None:
            continue
        grouped.setdefault(signature, []).append(str(node_id))

    rows: list[dict[str, Any]] = []
    for signature, members in sorted(grouped.items()):
        papers = _member_paper_ids(graph, members)
        if len(papers) < 2:
            continue
        members = sorted(members)
        subject, relation, object_value = signature
        hub_id = f"corpus::pattern::{_stable_id(*signature)}"
        graph.add_node(
            hub_id,
            type="CorpusPattern",
            label=f"{subject} {relation} {object_value}",
            pattern_subject=subject,
            pattern_relation=relation,
            pattern_object=object_value,
            source_member_ids_json=json.dumps(members, ensure_ascii=False),
            source_paper_ids_json=json.dumps(papers, ensure_ascii=False),
            support_count=len(members),
            graph_layer="corpus_alignment",
            evidence_status="derived_corpus_alignment",
            requires_verification=False,
            corpus_node_kind="alignment_hub",
            node_text=(
                "type: CorpusPattern\n"
                f"pattern: {subject} {relation} {object_value}\n"
                f"papers: {', '.join(papers)}"
            ),
        )
        forward = _alignment_edge_attrs(
            relation="EXPRESSES_CORPUS_PATTERN",
            members=members,
            paper_ids=papers,
            derivation_rule="exact_confirmed_pattern_signature",
        )
        reverse = _alignment_edge_attrs(
            relation="HAS_PATTERN_MENTION",
            members=members,
            paper_ids=papers,
            derivation_rule="exact_confirmed_pattern_signature",
        )
        for member in members:
            graph.add_edge(
                member,
                hub_id,
                key=f"pattern:{_stable_id(member, hub_id, 'forward')}",
                **forward,
            )
            graph.add_edge(
                hub_id,
                member,
                key=f"pattern:{_stable_id(hub_id, member, 'reverse')}",
                **reverse,
            )
        rows.append({
            "hub_id": hub_id,
            "alignment_type": "confirmed_pattern",
            "pattern_subject": subject,
            "pattern_relation": relation,
            "pattern_object": object_value,
            "member_ids": members,
            "paper_ids": papers,
            "member_count": len(members),
        })
    return rows


def generate_cross_paper_review_candidates(
    graph: nx.MultiDiGraph,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for node_id, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("type", ""))
        if node_type not in _REVIEW_CANDIDATE_TYPES:
            continue
        normalized = normalize_scientific_text(
            _node_label(str(node_id), dict(attrs))
        )
        if not normalized:
            continue
        grouped.setdefault((node_type, normalized), []).append(str(node_id))

    rows: list[dict[str, Any]] = []
    for (node_type, normalized), members in sorted(grouped.items()):
        for left, right in itertools.combinations(sorted(members), 2):
            left_paper = str(graph.nodes[left].get("source_paper_id", ""))
            right_paper = str(graph.nodes[right].get("source_paper_id", ""))
            if not left_paper or not right_paper or left_paper == right_paper:
                continue
            left_label = _node_label(left, dict(graph.nodes[left]))
            right_label = _node_label(right, dict(graph.nodes[right]))
            rows.append({
                "candidate_id": (
                    "corpus_resolution:exact_label:"
                    f"{_stable_id(left, right, node_type, normalized)}"
                ),
                "decision": "unreviewed",
                "approved": False,
                "recommendation": "same_entity",
                "merge_safety": "review_required",
                "review_priority": (
                    "high"
                    if node_type in {
                        "Catalyst",
                        "CatalystModel",
                        "Support",
                        "Material",
                    }
                    else "normal"
                ),
                "node_type": node_type,
                "left_id": left,
                "right_id": right,
                "left_paper_id": left_paper,
                "right_paper_id": right_paper,
                "left_source_node_id": str(
                    graph.nodes[left].get("source_node_id", "")
                ),
                "right_source_node_id": str(
                    graph.nodes[right].get("source_node_id", "")
                ),
                "left_label": left_label,
                "right_label": right_label,
                "normalized_label": normalized,
                "reason": (
                    "Exact normalized label and node type match across papers; "
                    "no destructive merge is applied automatically."
                ),
            })
    return rows


def _remap_node_row(
    row: dict[str, Any],
    *,
    paper_id: str,
    id_map: dict[str, str],
) -> dict[str, Any]:
    remapped = dict(row)
    raw_node_id = str(remapped.get("node_id", ""))
    if raw_node_id:
        remapped["source_node_id"] = raw_node_id
        remapped["node_id"] = id_map.get(raw_node_id, raw_node_id)
    remapped["source_paper_id"] = paper_id
    remapped["source_paper_ids_json"] = _paper_ids_json(
        remapped.get("source_paper_ids_json"),
        paper_id,
    )
    return remapped


def _remap_evidence_row(
    row: dict[str, Any],
    *,
    paper_id: str,
    id_map: dict[str, str],
) -> dict[str, Any]:
    remapped = dict(row)
    for field in ("source", "target", "node_id"):
        if remapped.get(field) not in (None, ""):
            raw = str(remapped[field])
            remapped[field] = id_map.get(raw, raw)

    if remapped.get("projection_edge_id") not in (None, ""):
        raw_edge_id = str(remapped["projection_edge_id"])
        remapped["source_projection_edge_id"] = raw_edge_id
        remapped["projection_edge_id"] = namespace_edge_id(
            paper_id,
            raw_edge_id,
        )

    for field in (
        "source_edge_ids_json",
        "projection_edge_ids_json",
    ):
        if field in remapped:
            remapped[field] = _namespace_json_edge_id_list(
                remapped[field],
                paper_id,
            )

    for field in (
        "supporting_node_ids_json",
        "node_path_json",
    ):
        if field in remapped:
            remapped[field] = _remap_json_id_list(
                remapped[field],
                id_map,
            )

    remapped["source_paper_id"] = paper_id
    remapped["source_paper_ids_json"] = _paper_ids_json(
        remapped.get("source_paper_ids_json"),
        paper_id,
    )
    return remapped


def build_corpus_graph(
    bundles: list[ProjectionBundle],
    *,
    corpus_id: str,
    mode: ProjectionMode,
    add_registry_alignment: bool = True,
    add_pattern_alignment: bool = True,
) -> tuple[
    nx.MultiDiGraph,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not bundles:
        raise ValueError("At least one projection bundle is required.")
    paper_ids = [bundle.paper_id for bundle in bundles]
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Duplicate paper_id in corpus input.")
    if any(bundle.mode != mode for bundle in bundles):
        raise ValueError("All projection bundles must use the same mode.")

    corpus = nx.MultiDiGraph(
        graph_stage="corpus_projection",
        corpus_id=corpus_id,
        projection_mode=mode,
        corpus_policy="non_destructive_namespacing_plus_alignment_hubs_v1",
    )
    node_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    source_node_count = 0
    source_edge_count = 0
    per_paper: list[dict[str, Any]] = []

    for bundle in bundles:
        namespaced, id_map = namespace_projection(bundle)
        source_node_count += namespaced.number_of_nodes()
        source_edge_count += namespaced.number_of_edges()

        corpus.add_nodes_from(namespaced.nodes(data=True))
        corpus.add_edges_from(namespaced.edges(keys=True, data=True))

        node_rows.extend(
            _remap_node_row(row, paper_id=bundle.paper_id, id_map=id_map)
            for row in bundle.node_rows
        )
        evidence_rows.extend(
            _remap_evidence_row(row, paper_id=bundle.paper_id, id_map=id_map)
            for row in bundle.evidence_rows
        )

        per_paper.append({
            "paper_id": bundle.paper_id,
            "mode": bundle.mode,
            "nodes": bundle.graph.number_of_nodes(),
            "edges": bundle.graph.number_of_edges(),
            "graphml": str(bundle.graph_path),
            "node_text": str(bundle.node_text_path),
            "edge_evidence": str(bundle.edge_evidence_path),
            "summary": str(bundle.summary_path),
            "sha256": bundle.sha256,
            "bridge_extraction_id": str(
                bundle.summary.get("bridge_extraction_id", "")
            ),
            "bridge_policy_run_id": str(
                bundle.summary.get("bridge_policy_run_id", "")
            ),
            "candidate_bridge_policy_run_id": str(
                bundle.summary.get("candidate_bridge_policy_run_id", "")
            ),
        })

    registry_rows = (
        add_registry_alignment_hubs(corpus)
        if add_registry_alignment
        else []
    )
    pattern_rows = (
        add_pattern_alignment_hubs(corpus)
        if add_pattern_alignment
        else []
    )
    candidate_rows = generate_cross_paper_review_candidates(corpus)

    # Alignment edges are derived corpus edges, but they still receive stable
    # edge IDs and evidence-sidecar rows so corpus traversal can audit every
    # graph edge uniformly.
    for left, right, key, attrs in corpus.edges(keys=True, data=True):
        if str(attrs.get("corpus_edge_kind", "")) != "alignment":
            continue
        edge_id = str(
            attrs.get("edge_id")
            or f"corpus_projection:{_stable_id(left, right, key, attrs.get('relation', ''))}"
        )
        attrs["edge_id"] = edge_id
        attrs["projection_edge_ids_json"] = json.dumps(
            [edge_id],
            ensure_ascii=False,
        )
        evidence_rows.append({
            "projection_edge_id": edge_id,
            "source": str(left),
            "target": str(right),
            "relation": str(attrs.get("relation", "")),
            "evidence_status": "derived_corpus_alignment",
            "graph_layer": "corpus_alignment",
            "source_edge_ids_json": "[]",
            "supporting_node_ids_json": str(
                attrs.get("supporting_node_ids_json", "[]")
            ),
            "evidence_pointers_json": "[]",
            "derivation_rule": str(attrs.get("derivation_rule", "")),
            "source_paper_ids_json": str(
                attrs.get("source_paper_ids_json", "[]")
            ),
            "requires_verification": False,
        })

    # Append searchable text rows for the new corpus-level alignment hubs.
    existing_node_row_ids = {
        str(row.get("node_id", "")) for row in node_rows
    }
    for node_id, attrs in corpus.nodes(data=True):
        if str(node_id) in existing_node_row_ids:
            continue
        if str(attrs.get("corpus_node_kind", "")) != "alignment_hub":
            continue
        node_rows.append({
            "node_id": str(node_id),
            "type": str(attrs.get("type", "")),
            "label": _node_label(str(node_id), dict(attrs)),
            "node_text": str(attrs.get("node_text", "")),
            "graph_layer": str(attrs.get("graph_layer", "")),
            "evidence_status": str(attrs.get("evidence_status", "")),
            "source_paper_ids_json": str(
                attrs.get("source_paper_ids_json", "[]")
            ),
            "requires_verification": bool(
                attrs.get("requires_verification", False)
            ),
        })

    manifest = {
        "corpus_id": corpus_id,
        "mode": mode,
        "policy_version": (
            "corpus-non-destructive-alignment-v1"
        ),
        "paper_ids": paper_ids,
        "paper_count": len(paper_ids),
        "papers": per_paper,
        "source_projection_nodes": source_node_count,
        "source_projection_edges": source_edge_count,
        "corpus_nodes": corpus.number_of_nodes(),
        "corpus_edges": corpus.number_of_edges(),
        "registry_alignment_hubs": len(registry_rows),
        "pattern_alignment_hubs": len(pattern_rows),
        "cross_paper_review_candidates": len(candidate_rows),
        "destructive_cross_paper_merges": 0,
    }

    return (
        corpus,
        node_rows,
        evidence_rows,
        registry_rows,
        pattern_rows,
        candidate_rows,
        manifest,
    )


def audit_corpus_graph(
    graph: nx.MultiDiGraph,
    *,
    expected_papers: Iterable[str],
    expected_source_nodes: int | None = None,
    expected_source_edges: int | None = None,
) -> dict[str, Any]:
    expected = sorted(set(map(str, expected_papers)))
    issues: list[dict[str, Any]] = []

    paper_local_nodes = [
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("corpus_node_kind", "")) == "paper_local"
    ]
    alignment_hubs = [
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("corpus_node_kind", "")) == "alignment_hub"
    ]
    source_edges = [
        (str(left), str(right), str(key), dict(attrs))
        for left, right, key, attrs in graph.edges(keys=True, data=True)
        if str(attrs.get("corpus_edge_kind", "")) != "alignment"
    ]
    alignment_edges = [
        (str(left), str(right), str(key), dict(attrs))
        for left, right, key, attrs in graph.edges(keys=True, data=True)
        if str(attrs.get("corpus_edge_kind", "")) == "alignment"
    ]

    if expected_source_nodes is not None and len(paper_local_nodes) != expected_source_nodes:
        issues.append({
            "issue": "source_node_count_mismatch",
            "expected": expected_source_nodes,
            "actual": len(paper_local_nodes),
        })
    if expected_source_edges is not None and len(source_edges) != expected_source_edges:
        issues.append({
            "issue": "source_edge_count_mismatch",
            "expected": expected_source_edges,
            "actual": len(source_edges),
        })

    seen_papers = sorted({
        str(graph.nodes[node_id].get("source_paper_id", ""))
        for node_id in paper_local_nodes
        if str(graph.nodes[node_id].get("source_paper_id", "")).strip()
    })
    if seen_papers != expected:
        issues.append({
            "issue": "paper_set_mismatch",
            "expected": expected,
            "actual": seen_papers,
        })

    for node_id in paper_local_nodes:
        attrs = graph.nodes[node_id]
        paper_id = str(attrs.get("source_paper_id", ""))
        source_node_id = str(attrs.get("source_node_id", ""))
        expected_prefix = f"paper::{paper_id}::"
        if not paper_id or not source_node_id or not node_id.startswith(expected_prefix):
            issues.append({
                "issue": "invalid_paper_local_identity",
                "node_id": node_id,
                "source_paper_id": paper_id,
                "source_node_id": source_node_id,
            })

    direct_cross_paper_edges = 0
    for left, right, key, attrs in source_edges:
        left_paper = str(graph.nodes[left].get("source_paper_id", ""))
        right_paper = str(graph.nodes[right].get("source_paper_id", ""))
        if left_paper and right_paper and left_paper != right_paper:
            direct_cross_paper_edges += 1
            issues.append({
                "issue": "unexpected_direct_cross_paper_source_edge",
                "source": left,
                "target": right,
                "key": key,
                "source_paper": left_paper,
                "target_paper": right_paper,
            })
        if not str(attrs.get("source_paper_id", "")).strip():
            issues.append({
                "issue": "source_edge_missing_source_paper_id",
                "source": left,
                "target": right,
                "key": key,
            })

    for hub_id in alignment_hubs:
        attrs = graph.nodes[hub_id]
        papers = [str(item) for item in _json_list(attrs.get("source_paper_ids_json"))]
        members = [str(item) for item in _json_list(attrs.get("source_member_ids_json"))]
        if len(set(papers)) < 2 or len(members) < 2:
            issues.append({
                "issue": "invalid_alignment_hub_support",
                "hub_id": hub_id,
                "paper_ids": papers,
                "member_ids": members,
            })
        missing_members = [member for member in members if member not in graph]
        if missing_members:
            issues.append({
                "issue": "alignment_hub_missing_member",
                "hub_id": hub_id,
                "missing_members": missing_members,
            })

    return {
        "graph_stage": str(graph.graph.get("graph_stage", "")),
        "corpus_id": str(graph.graph.get("corpus_id", "")),
        "projection_mode": str(graph.graph.get("projection_mode", "")),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "paper_local_nodes": len(paper_local_nodes),
        "source_projection_edges": len(source_edges),
        "alignment_hubs": len(alignment_hubs),
        "alignment_edges": len(alignment_edges),
        "direct_cross_paper_source_edges": direct_cross_paper_edges,
        "expected_papers": expected,
        "seen_papers": seen_papers,
        "issues": issues,
        "issue_count": len(issues),
        "passes_structural_gate": len(issues) == 0,
        "destructive_cross_paper_merges": 0,
    }
