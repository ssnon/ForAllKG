from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from dac_her.explorer_contracts import (
    AlignmentContext,
    CorpusScope,
    EdgeEvidence,
    EndpointView,
    EvidenceCatalog,
    ExplorerDirectHit,
    ExplorerPath,
    ExplorerPathQuality,
    ExplorerPolicy,
    ExplorerStep,
    ExplorerTask,
    GraphExplorerPacket,
    PaperScope,
    ProvenanceSummary,
    RetrievalSummary,
    WaypointView,
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def _jsonish(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _jsonish_strings(value: Any) -> list[str]:
    return sorted({str(item) for item in _jsonish(value) if str(item).strip()})


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _parse_scientific_direction(step: dict[str, Any]) -> tuple[str, str]:
    direction = str(step.get("scientific_direction", "")).strip()
    if " -> " in direction:
        left, right = direction.split(" -> ", 1)
        return left.strip(), right.strip()
    source = str(step.get("source", ""))
    target = str(step.get("target", ""))
    if str(step.get("traversal_direction", "forward")) == "reverse":
        return target, source
    return source, target


def _canonical_item_key(value: Any) -> str:
    if isinstance(value, str):
        return "str:" + value
    return "json:" + json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _merge_jsonish_items(*values: Any) -> list[dict[str, Any] | str]:
    merged: dict[str, dict[str, Any] | str] = {}
    for value in values:
        for item in _jsonish(value):
            if not isinstance(item, (dict, str)):
                continue
            merged.setdefault(_canonical_item_key(item), item)
    return [merged[key] for key in sorted(merged)]


def _selected_alternative(
    step: dict[str, Any],
    original_edge_id: str,
) -> dict[str, Any] | None:
    alternatives = step.get("alternatives")
    if not isinstance(alternatives, list):
        return None
    matches = [
        dict(item)
        for item in alternatives
        if isinstance(item, dict)
        and str(item.get("original_edge_id", "")).strip() == original_edge_id
    ]
    if not matches:
        return None
    scientific_source, scientific_target = _parse_scientific_direction(step)
    relation = str(step.get("relation", ""))
    exact = [
        item
        for item in matches
        if str(item.get("original_source", "")) == scientific_source
        and str(item.get("original_target", "")) == scientific_target
        and str(item.get("relation", "")) == relation
    ]
    candidates = exact or matches
    candidates.sort(
        key=lambda item: (
            str(item.get("original_source", "")),
            str(item.get("relation", "")),
            str(item.get("original_target", "")),
            str(item.get("original_edge_key", "")),
        )
    )
    if len(candidates) > 1:
        signatures = {
            (
                str(item.get("original_source", "")),
                str(item.get("relation", "")),
                str(item.get("original_target", "")),
                str(item.get("evidence_pointers_json", "")),
            )
            for item in candidates
        }
        if len(signatures) > 1:
            raise ValueError(
                f"Ambiguous traversal alternatives for selected edge {original_edge_id!r}."
            )
    return candidates[0]


def _is_alignment_edge(
    *,
    edge_class: str = "",
    evidence_row: dict[str, Any] | None = None,
) -> bool:
    row = evidence_row or {}
    return (
        edge_class in {"registry_alignment", "pattern_alignment"}
        or str(row.get("graph_layer", "")) == "corpus_alignment"
        or str(row.get("evidence_status", "")) == "derived_corpus_alignment"
    )


def _alignment_hub(
    step: dict[str, Any],
    node_index: dict[str, dict[str, Any]],
) -> str | None:
    candidates = [
        str(step.get("source", "")),
        str(step.get("target", "")),
    ]
    for node_id in candidates:
        if node_id.startswith("corpus::"):
            return node_id
        row = node_index.get(node_id, {})
        if str(row.get("type") or row.get("node_type") or "") in {
            "CorpusAlignment",
            "CorpusPattern",
        }:
            return node_id
    return None


def _edge_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for row in rows:
        keys: set[str] = set()
        for field in (
            "projection_edge_id",
            "edge_id",
            "source_projection_edge_id",
            "source_edge_id",
        ):
            value = str(row.get(field, "")).strip()
            if value:
                keys.add(value)
        keys.update(_jsonish_strings(row.get("projection_edge_ids_json")))
        for key in keys:
            previous = index.get(key)
            if previous is None:
                index[key] = row
                continue
            signature_previous = (
                str(previous.get("source", "")),
                str(previous.get("relation", "")),
                str(previous.get("target", "")),
            )
            signature_new = (
                str(row.get("source", "")),
                str(row.get("relation", "")),
                str(row.get("target", "")),
            )
            if signature_previous != signature_new:
                ambiguous.add(key)
    if ambiguous:
        raise ValueError(
            "Ambiguous edge-evidence IDs: " + ", ".join(sorted(ambiguous)[:10])
        )
    return index


def _paper_id_from_node(node_id: str) -> str | None:
    if not node_id.startswith("paper::"):
        return None
    parts = node_id.split("::", 2)
    if len(parts) != 3:
        return None
    return parts[1] or None


def _paper_scope(manifest: dict[str, Any]) -> tuple[list[PaperScope], dict[str, PaperScope]]:
    papers: list[PaperScope] = []
    for row in manifest.get("papers", []):
        if not isinstance(row, dict):
            continue
        paper_id = str(row.get("paper_id", "")).strip()
        if not paper_id:
            continue
        quality = str(row.get("extraction_quality_status", "unknown") or "unknown")
        if quality not in {
            "complete",
            "partial_acceptable",
            "partial_critical",
            "rejected",
            "unknown",
        }:
            quality = "unknown"
        coverage = _as_float(row.get("extraction_source_token_coverage"))
        quarantine = _as_float(row.get("extraction_quarantine_token_fraction"))
        papers.append(
            PaperScope(
                paper_id=paper_id,
                quality_status=quality,
                source_token_coverage=coverage,
                quarantine_token_fraction=quarantine,
                absence_claims_allowed=_as_bool(
                    row.get("extraction_absence_claims_allowed", False)
                ),
            )
        )
    papers.sort(key=lambda item: item.paper_id)
    return papers, {paper.paper_id: paper for paper in papers}


def _question_from_payload(payload: dict[str, Any]) -> str:
    source = payload.get("source_query") or "the source concept"
    target = payload.get("target_query") or "the target concept"
    waypoint = payload.get("semantic_stop_query")
    if waypoint:
        return f"How does {source} connect to {target} through {waypoint}?"
    return f"How does {source} connect to {target} in the supplied evidence graph?"


def _endpoint_view(path: dict[str, Any]) -> EndpointView:
    pair = path.get("endpoint_pair")
    source_match = path.get("source_match") if isinstance(path.get("source_match"), dict) else {}
    target_match = path.get("target_match") if isinstance(path.get("target_match"), dict) else {}
    if not isinstance(pair, dict):
        pair = {}
    return EndpointView(
        source_node_id=str(pair.get("source_node_id") or source_match.get("node_id") or path.get("source") or ""),
        target_node_id=str(pair.get("target_node_id") or target_match.get("node_id") or path.get("target") or ""),
        source_label=(
            str(pair.get("source_label") or source_match.get("label"))
            if pair.get("source_label") or source_match.get("label")
            else None
        ),
        target_label=(
            str(pair.get("target_label") or target_match.get("label"))
            if pair.get("target_label") or target_match.get("label")
            else None
        ),
        source_similarity=_as_float(pair.get("source_similarity", source_match.get("semantic_similarity"))),
        target_similarity=_as_float(pair.get("target_similarity", target_match.get("semantic_similarity"))),
        semantic_tier=(int(pair["semantic_tier"]) if pair.get("semantic_tier") is not None else None),
        pair_score=_as_float(pair.get("pair_score")),
        source_exact=(bool(pair.get("source_exact")) if pair.get("source_exact") is not None else None),
        target_exact=(bool(pair.get("target_exact")) if pair.get("target_exact") is not None else None),
    )


def _waypoint_view(path: dict[str, Any]) -> WaypointView | None:
    waypoint = path.get("waypoint")
    stop_match = path.get("stop_match") if isinstance(path.get("stop_match"), dict) else {}
    if not isinstance(waypoint, dict):
        return None
    node_id = str(waypoint.get("node_id") or stop_match.get("node_id") or path.get("semantic_stop") or "")
    if not node_id:
        return None
    return WaypointView(
        node_id=node_id,
        label=(str(waypoint.get("label") or stop_match.get("label")) if waypoint.get("label") or stop_match.get("label") else None),
        semantic_tier=(int(waypoint["semantic_tier"]) if waypoint.get("semantic_tier") is not None else None),
        semantic_similarity=_as_float(waypoint.get("semantic_similarity", stop_match.get("semantic_similarity"))),
        waypoint_rank=(int(waypoint["waypoint_rank"]) if waypoint.get("waypoint_rank") is not None else None),
    )


def _path_quality(path: dict[str, Any]) -> ExplorerPathQuality:
    row = path.get("path_quality") if isinstance(path.get("path_quality"), dict) else {}
    return ExplorerPathQuality(
        path_type=str(row.get("path_type", "UNKNOWN")),
        path_structure_type=(str(row["path_structure_type"]) if row.get("path_structure_type") is not None else None),
        path_tags=[str(item) for item in row.get("path_tags", [])],
        endpoint_semantic_tier=(int(row["endpoint_semantic_tier"]) if row.get("endpoint_semantic_tier") is not None else None),
        endpoint_pair_score=_as_float(row.get("endpoint_pair_score")),
        mechanism_edge_count=int(row.get("mechanism_edge_count", 0) or 0),
        mechanism_node_count=int(row.get("mechanism_node_count", 0) or 0),
        mechanism_node_ids=[str(item) for item in row.get("mechanism_node_ids", [])],
        mechanistic_content=(str(row["mechanistic_content"]) if row.get("mechanistic_content") is not None else None),
        mechanistic_content_basis=(str(row["mechanistic_content_basis"]) if row.get("mechanistic_content_basis") is not None else None),
        mechanism_bearing=_as_bool(row.get("mechanism_bearing", False)),
        navigation_edge_fraction=float(row.get("navigation_edge_fraction", 0.0) or 0.0),
        reverse_fraction=float(row.get("reverse_fraction", 0.0) or 0.0),
        candidate_fraction=float(row.get("candidate_fraction", 0.0) or 0.0),
        endpoint_relevance=(str(row["endpoint_relevance"]) if row.get("endpoint_relevance") is not None else None),
        navigation_burden=(str(row["navigation_burden"]) if row.get("navigation_burden") is not None else None),
        reverse_burden=(str(row["reverse_burden"]) if row.get("reverse_burden") is not None else None),
        visited_paper_count=int(row.get("visited_paper_count", path.get("visited_paper_count", 0)) or 0),
        shared_entity_bridge=_as_bool(row.get("shared_entity_bridge", False)),
    )


class GraphExplorerPacketBuilder:
    def __init__(
        self,
        *,
        substrate_version: str = "traversal-substrate-v2.4.7",
        strict_provenance: bool = True,
    ) -> None:
        self.substrate_version = substrate_version
        self.strict_provenance = bool(strict_provenance)

    def build_from_files(
        self,
        *,
        traversal_result_path: str | Path,
        corpus_dir: str | Path,
        question: str | None = None,
        objective: str = "map_evidence",
    ) -> GraphExplorerPacket:
        corpus_dir = Path(corpus_dir)
        return self.build(
            traversal_payload=_read_json(traversal_result_path),
            node_rows=_read_jsonl(corpus_dir / "node_text.jsonl"),
            edge_rows=_read_jsonl(corpus_dir / "edge_evidence.jsonl"),
            corpus_manifest=_read_json(corpus_dir / "manifest.json"),
            question=question,
            objective=objective,
        )

    def build(
        self,
        *,
        traversal_payload: dict[str, Any],
        node_rows: list[dict[str, Any]],
        edge_rows: list[dict[str, Any]],
        corpus_manifest: dict[str, Any],
        question: str | None = None,
        objective: str = "map_evidence",
    ) -> GraphExplorerPacket:
        corpus_id = str(traversal_payload.get("corpus_id") or corpus_manifest.get("corpus_id") or "").strip()
        mode = str(traversal_payload.get("mode") or corpus_manifest.get("mode") or "").strip()
        if not corpus_id:
            raise ValueError("Missing corpus_id in traversal payload/manifest.")
        if mode not in {"evidence", "mechanism", "exploratory"}:
            raise ValueError(f"Unsupported traversal mode: {mode!r}")

        papers, paper_by_id = _paper_scope(corpus_manifest)
        node_index = {
            str(row.get("node_id", "")): row
            for row in node_rows
            if str(row.get("node_id", "")).strip()
        }
        edge_index = _edge_index(edge_rows)
        traversal_evidence_by_edge: dict[str, dict[str, Any]] = {}
        edge_class_by_id: dict[str, str] = {}
        alignment_context_acc: dict[tuple[str, str], dict[str, set[str]]] = {}
        alignment_member_node_ids: set[str] = set()

        task_question = question or _question_from_payload(traversal_payload)
        task_id = _stable_id(
            "task",
            corpus_id,
            mode,
            traversal_payload.get("source_query") or "",
            traversal_payload.get("target_query") or "",
            traversal_payload.get("semantic_stop_query") or "",
            objective,
            task_question,
        )
        task = ExplorerTask(
            task_id=task_id,
            question=task_question,
            source_query=(str(traversal_payload["source_query"]) if traversal_payload.get("source_query") is not None else None),
            target_query=(str(traversal_payload["target_query"]) if traversal_payload.get("target_query") is not None else None),
            waypoint_query=(str(traversal_payload["semantic_stop_query"]) if traversal_payload.get("semantic_stop_query") is not None else None),
            traversal_mode=mode,
            objective=objective,
        )

        direct_hits: list[ExplorerDirectHit] = []
        referenced_node_ids: set[str] = set()
        for hit in traversal_payload.get("direct_concept_hits", []):
            if not isinstance(hit, dict):
                continue
            node_id = str(hit.get("node_id", "")).strip()
            if not node_id:
                continue
            referenced_node_ids.add(node_id)
            direct_hits.append(
                ExplorerDirectHit(
                    hit_id=_stable_id("hit", task_id, node_id, hit.get("quality_basis", "")),
                    node_id=node_id,
                    node_evidence_ref=node_id,
                    hit_tier=int(hit.get("hit_tier", 99)),
                    quality_basis=str(hit.get("quality_basis", "")),
                    source_similarity=_as_float(hit.get("source_similarity")),
                    target_similarity=_as_float(hit.get("target_similarity")),
                    mechanism_bearing=_as_bool(hit.get("mechanism_bearing", False)),
                    requires_verification=_as_bool(hit.get("requires_verification", False)),
                )
            )

        explorer_paths: list[ExplorerPath] = []
        referenced_edge_ids: set[str] = set()
        for bundle_rank, path in enumerate(traversal_payload.get("paths", []), start=1):
            if not isinstance(path, dict):
                continue
            steps: list[ExplorerStep] = []
            path_nodes = [str(item) for item in path.get("nodes", [])]
            referenced_node_ids.update(path_nodes)
            for step in path.get("steps", []):
                if not isinstance(step, dict):
                    continue
                original_edge_id = str(step.get("selected_original_edge_id", "")).strip()
                if not original_edge_id:
                    if self.strict_provenance:
                        raise ValueError(f"Path {path.get('path_id')} has a step without selected_original_edge_id.")
                    original_edge_id = str(step.get("navigation_edge_id", "")).strip()
                evidence_row = edge_index.get(original_edge_id)
                if evidence_row is None and self.strict_provenance:
                    raise KeyError(
                        f"Missing edge evidence for selected_original_edge_id={original_edge_id!r} "
                        f"in path {path.get('path_id')!r}."
                    )
                scientific_source, scientific_target = _parse_scientific_direction(step)
                if evidence_row is not None:
                    row_source = str(evidence_row.get("source", "")).strip()
                    row_target = str(evidence_row.get("target", "")).strip()
                    row_relation = str(evidence_row.get("relation", "")).strip()
                    if self.strict_provenance and row_source and row_source != scientific_source:
                        raise ValueError(
                            f"Scientific source mismatch for {original_edge_id}: "
                            f"step={scientific_source!r}, evidence={row_source!r}"
                        )
                    if self.strict_provenance and row_target and row_target != scientific_target:
                        raise ValueError(
                            f"Scientific target mismatch for {original_edge_id}: "
                            f"step={scientific_target!r}, evidence={row_target!r}"
                        )
                    if self.strict_provenance and row_relation and row_relation != str(step.get("relation", "")):
                        raise ValueError(
                            f"Relation mismatch for {original_edge_id}: "
                            f"step={step.get('relation')!r}, evidence={row_relation!r}"
                        )

                selected_alternative = _selected_alternative(step, original_edge_id)
                if selected_alternative is not None:
                    previous_alternative = traversal_evidence_by_edge.get(original_edge_id)
                    if previous_alternative is not None:
                        previous_signature = (
                            str(previous_alternative.get("original_source", "")),
                            str(previous_alternative.get("relation", "")),
                            str(previous_alternative.get("original_target", "")),
                            str(previous_alternative.get("evidence_pointers_json", "")),
                        )
                        current_signature = (
                            str(selected_alternative.get("original_source", "")),
                            str(selected_alternative.get("relation", "")),
                            str(selected_alternative.get("original_target", "")),
                            str(selected_alternative.get("evidence_pointers_json", "")),
                        )
                        if previous_signature != current_signature:
                            raise ValueError(
                                f"Conflicting traversal provenance for repeated edge {original_edge_id!r}."
                            )
                    else:
                        traversal_evidence_by_edge[original_edge_id] = selected_alternative

                edge_class = str(step.get("edge_class", ""))
                previous_class = edge_class_by_id.get(original_edge_id)
                if previous_class and previous_class != edge_class:
                    raise ValueError(
                        f"Conflicting edge classes for repeated edge {original_edge_id!r}: "
                        f"{previous_class!r} != {edge_class!r}"
                    )
                edge_class_by_id[original_edge_id] = edge_class

                alignment_edge = _is_alignment_edge(
                    edge_class=edge_class,
                    evidence_row=evidence_row,
                )
                supporting_ids = set(
                    _jsonish_strings(
                        (evidence_row or {}).get("supporting_node_ids_json")
                    )
                )
                if selected_alternative is not None:
                    supporting_ids.update(
                        _jsonish_strings(
                            selected_alternative.get("supporting_node_ids_json")
                        )
                    )

                if alignment_edge:
                    alignment_member_node_ids.update(supporting_ids)
                    hub_id = _alignment_hub(step, node_index)
                    if hub_id:
                        key = (str(path.get("path_id", "")), hub_id)
                        context = alignment_context_acc.setdefault(
                            key,
                            {
                                "alignment_edge_ids": set(),
                                "member_node_ids": set(),
                                "member_paper_ids": set(),
                                "traversed_entry_node_ids": set(),
                                "traversed_exit_node_ids": set(),
                            },
                        )
                        context["alignment_edge_ids"].add(original_edge_id)
                        context["member_node_ids"].update(supporting_ids)
                        context["member_paper_ids"].update(
                            _jsonish_strings(
                                (evidence_row or {}).get("source_paper_ids_json")
                            )
                        )
                        if selected_alternative is not None:
                            context["member_paper_ids"].update(
                                _jsonish_strings(
                                    selected_alternative.get("source_paper_ids")
                                )
                            )
                        nav_source = str(step.get("source", ""))
                        nav_target = str(step.get("target", ""))
                        if nav_target == hub_id and nav_source:
                            context["traversed_entry_node_ids"].add(nav_source)
                        if nav_source == hub_id and nav_target:
                            context["traversed_exit_node_ids"].add(nav_target)
                else:
                    # Non-alignment supporting nodes can carry actual evidence
                    # needed to interpret a projected/derived scientific edge.
                    # Materialize those nodes in the agent-facing catalog.
                    referenced_node_ids.update(supporting_ids)

                referenced_edge_ids.add(original_edge_id)
                steps.append(
                    ExplorerStep(
                        navigation_source=str(step.get("source", "")),
                        navigation_target=str(step.get("target", "")),
                        traversal_direction=str(step.get("traversal_direction", "forward")),
                        scientific_source=scientific_source,
                        relation=str(step.get("relation", "")),
                        scientific_target=scientific_target,
                        selected_original_edge_id=original_edge_id,
                        edge_evidence_ref=original_edge_id,
                        edge_class=str(step.get("edge_class", "")),
                        requires_verification=_as_bool(step.get("requires_verification", False)),
                    )
                )
            explorer_paths.append(
                ExplorerPath(
                    path_id=str(path.get("path_id", "")),
                    bundle_rank=bundle_rank,
                    endpoint=_endpoint_view(path),
                    waypoint=_waypoint_view(path),
                    node_ids=path_nodes,
                    steps=steps,
                    visited_paper_ids=sorted({str(item) for item in path.get("visited_paper_ids", []) if str(item).strip()}),
                    supporting_paper_ids=sorted({str(item) for item in path.get("supporting_paper_ids", []) if str(item).strip()}),
                    hub_scope_paper_ids=sorted({str(item) for item in path.get("hub_scope_paper_ids", []) if str(item).strip()}),
                    quality=_path_quality(path),
                )
            )

        node_catalog = {}
        for node_id in sorted(referenced_node_ids):
            row = node_index.get(node_id)
            if row is None:
                if self.strict_provenance:
                    raise KeyError(f"Referenced node is missing from node_text.jsonl: {node_id}")
                row = {"node_id": node_id, "type": "Unknown", "label": node_id, "node_text": node_id}
            paper_id = str(row.get("source_paper_id", "")).strip() or _paper_id_from_node(node_id)
            scope = paper_by_id.get(paper_id or "")
            node_catalog[node_id] = {
                "node_id": node_id,
                "node_type": str(row.get("type") or row.get("node_type") or "Unknown"),
                "label": str(row.get("label") or row.get("statement") or row.get("name") or node_id),
                "node_text": str(row.get("node_text") or row.get("label") or row.get("statement") or row.get("name") or node_id),
                "graph_layer": str(row.get("graph_layer", "")),
                "evidence_status": str(row.get("evidence_status", "")),
                "requires_verification": _as_bool(row.get("requires_verification", False)),
                "source_paper_id": paper_id or None,
                "source_paper_ids": _jsonish_strings(row.get("source_paper_ids_json")) or ([paper_id] if paper_id else []),
                "extraction_quality_status": (scope.quality_status if scope else None),
                "absence_claims_allowed": (scope.absence_claims_allowed if scope else False),
            }

        edge_catalog: dict[str, EdgeEvidence] = {}
        recovered_pointer_edge_count = 0
        derived_alignment_edge_count = 0
        missing_pointer_edge_count = 0
        pointer_grounded_edge_count = 0
        for edge_id in sorted(referenced_edge_ids):
            row = edge_index.get(edge_id)
            if row is None:
                if self.strict_provenance:
                    raise KeyError(f"Referenced edge is missing from edge_evidence.jsonl: {edge_id}")
                continue
            fallback = traversal_evidence_by_edge.get(edge_id) or {}
            source = str(row.get("source") or fallback.get("original_source") or "")
            target = str(row.get("target") or fallback.get("original_target") or "")
            relation = str(row.get("relation") or fallback.get("relation") or "")
            graph_layer = str(row.get("graph_layer") or fallback.get("graph_layer") or "")
            evidence_status = str(row.get("evidence_status") or fallback.get("evidence_status") or "")

            paper_ids = set(_jsonish_strings(row.get("source_paper_ids_json")))
            direct = str(row.get("source_paper_id", "")).strip()
            if direct:
                paper_ids.add(direct)
            paper_ids.update(_jsonish_strings(fallback.get("source_paper_ids")))

            sidecar_pointers = _merge_jsonish_items(row.get("evidence_pointers_json"))
            traversal_pointers = _merge_jsonish_items(fallback.get("evidence_pointers_json"))
            evidence_pointers = _merge_jsonish_items(
                sidecar_pointers,
                traversal_pointers,
            )
            if sidecar_pointers and traversal_pointers:
                pointer_source = "edge_sidecar+traversal_selected_alternative"
            elif sidecar_pointers:
                pointer_source = "edge_sidecar"
            elif traversal_pointers:
                pointer_source = "traversal_selected_alternative"
                recovered_pointer_edge_count += 1
            else:
                pointer_source = "missing"

            supporting_node_ids = set(
                _jsonish_strings(row.get("supporting_node_ids_json"))
            )
            supporting_node_ids.update(
                _jsonish_strings(fallback.get("supporting_node_ids_json"))
            )
            derivation_rule = (
                str(row["derivation_rule"])
                if row.get("derivation_rule") not in (None, "")
                else None
            )
            alignment_edge = _is_alignment_edge(
                edge_class=edge_class_by_id.get(edge_id, ""),
                evidence_row=row,
            )

            if evidence_pointers:
                provenance_status = "grounded_pointer"
                pointer_grounded_edge_count += 1
            elif alignment_edge and derivation_rule and supporting_node_ids:
                provenance_status = "derived_alignment"
                pointer_source = "derived_alignment"
                derived_alignment_edge_count += 1
            else:
                provenance_status = "missing_pointer"
                missing_pointer_edge_count += 1
                if self.strict_provenance:
                    raise ValueError(
                        "Missing evidence pointer for non-alignment scientific edge "
                        f"{edge_id!r}. The edge sidecar and the traversal selected "
                        "alternative were both empty. Refusing to build an agent packet "
                        "with a broken provenance chain."
                    )

            edge_catalog[edge_id] = EdgeEvidence(
                edge_id=edge_id,
                scientific_source=source,
                relation=relation,
                scientific_target=target,
                graph_layer=graph_layer,
                evidence_status=evidence_status,
                requires_verification=(
                    _as_bool(row.get("requires_verification", False))
                    or _as_bool(fallback.get("requires_verification", False))
                ),
                source_paper_ids=sorted(paper_ids),
                evidence_pointers=evidence_pointers,
                supporting_node_ids=sorted(supporting_node_ids),
                derivation_rule=derivation_rule,
                evidence_pointer_source=pointer_source,
                provenance_status=provenance_status,
            )

        alignment_contexts: list[AlignmentContext] = []
        for (path_id, hub_id), values in sorted(alignment_context_acc.items()):
            hub_row = node_index.get(hub_id, {})
            alignment_contexts.append(
                AlignmentContext(
                    context_id=_stable_id("alignment", path_id, hub_id),
                    path_id=path_id,
                    hub_node_id=hub_id,
                    hub_label=(
                        str(hub_row.get("label") or hub_row.get("statement") or hub_row.get("name"))
                        if hub_row.get("label") or hub_row.get("statement") or hub_row.get("name")
                        else None
                    ),
                    hub_type=(
                        str(hub_row.get("type") or hub_row.get("node_type"))
                        if hub_row.get("type") or hub_row.get("node_type")
                        else None
                    ),
                    alignment_edge_ids=sorted(values["alignment_edge_ids"]),
                    member_node_ids=sorted(values["member_node_ids"]),
                    member_paper_ids=sorted(values["member_paper_ids"]),
                    traversed_entry_node_ids=sorted(values["traversed_entry_node_ids"]),
                    traversed_exit_node_ids=sorted(values["traversed_exit_node_ids"]),
                )
            )

        suppressed_alignment_member_node_count = len(
            alignment_member_node_ids - set(node_catalog)
        )
        provenance_summary = ProvenanceSummary(
            strict_provenance=self.strict_provenance,
            edge_count=len(edge_catalog),
            pointer_grounded_edge_count=pointer_grounded_edge_count,
            pointer_recovered_from_traversal_count=recovered_pointer_edge_count,
            derived_alignment_edge_count=derived_alignment_edge_count,
            missing_pointer_edge_count=missing_pointer_edge_count,
            materialized_node_count=len(node_catalog),
            suppressed_alignment_member_node_count=(
                suppressed_alignment_member_node_count
            ),
        )

        retrieval_summary = RetrievalSummary(
            algorithm=str(traversal_payload.get("algorithm", "")),
            effective_max_depth=(int(traversal_payload["effective_max_depth"]) if traversal_payload.get("effective_max_depth") is not None else None),
            direct_concept_hit_count=len(direct_hits),
            returned_path_count=len(explorer_paths),
            returned_path_type_counts={str(k): int(v) for k, v in dict(traversal_payload.get("returned_path_type_counts", traversal_payload.get("path_type_counts", {}))).items()},
            endpoint_selector_enabled=(bool(traversal_payload["endpoint_selector_enabled"]) if traversal_payload.get("endpoint_selector_enabled") is not None else None),
            waypoint_selector_enabled=(bool(traversal_payload["waypoint_selector_enabled"]) if traversal_payload.get("waypoint_selector_enabled") is not None else None),
            candidate_path_count=(int(traversal_payload["candidate_path_count"]) if traversal_payload.get("candidate_path_count") is not None else None),
        )

        packet_id = _stable_id(
            "packet",
            task_id,
            corpus_id,
            mode,
            *[path.path_id for path in explorer_paths],
            *[hit.hit_id for hit in direct_hits],
        )
        packet = GraphExplorerPacket(
            packet_id=packet_id,
            packet_sha256="",
            task=task,
            corpus=CorpusScope(
                corpus_id=corpus_id,
                projection_mode=mode,
                papers=papers,
                substrate_version=self.substrate_version,
            ),
            retrieval_summary=retrieval_summary,
            direct_concept_hits=direct_hits,
            paths=explorer_paths,
            evidence_catalog=EvidenceCatalog(nodes=node_catalog, edges=edge_catalog),
            alignment_contexts=alignment_contexts,
            provenance_summary=provenance_summary,
            policy=ExplorerPolicy(),
        )
        payload_for_hash = packet.model_dump(mode="json")
        payload_for_hash["packet_sha256"] = ""
        packet_sha = hashlib.sha256(_canonical_json(payload_for_hash)).hexdigest()
        return packet.model_copy(update={"packet_sha256": packet_sha})


def write_packet(packet: GraphExplorerPacket, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
