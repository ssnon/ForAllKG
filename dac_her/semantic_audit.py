from __future__ import annotations

import itertools
import json
import re
from typing import Any, Iterable, Sequence

import networkx as nx

from dac_her.claim_overlap import ClaimOverlapCandidate
from dac_her.graph_normalization import refine_semantic_metric_id
from dac_her.resolution_candidates import normalize_scientific_text
from dac_her.semantic_repairs import node_composition_signature


_VISUAL_LOCATOR_RE = re.compile(
    r"\b(?:(?:supplementary|supplemental)\s+)?"
    r"(?:figs?|figures?|schemes?)\.?\s*S?\d+[A-Za-z]?\b",
    re.IGNORECASE,
)
_TABLE_LOCATOR_RE = re.compile(
    r"\b(?:(?:supplementary|supplemental)\s+)?"
    r"tables?\.?\s*S?\d+[A-Za-z]?\b",
    re.IGNORECASE,
)
_CROSS_DOCUMENT_TYPES = {
    "Catalyst",
    "CatalystModel",
    "Support",
    "CoordinationMotif",
    "Metal",
    "Reaction",
    "SynthesisMethod",
    "Precursor",
    "Material",
}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _edge_iter(graph: nx.Graph):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for index, (source, target, data) in enumerate(graph.edges(data=True)):
            yield source, target, str(index), data


def model_of_composition_issues(graph: nx.Graph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target, key, data in _edge_iter(graph):
        if str(data.get("relation", "")) != "MODEL_OF":
            continue
        model_signature = node_composition_signature(graph, str(source))
        catalyst_signature = node_composition_signature(graph, str(target))
        if not model_signature or not catalyst_signature or model_signature == catalyst_signature:
            continue
        rows.append({
            "source": str(source),
            "target": str(target),
            "edge_key": str(key),
            "model_label": graph.nodes[source].get("label", ""),
            "catalyst_label": graph.nodes[target].get("label", ""),
            "model_composition": dict(model_signature),
            "catalyst_composition": dict(catalyst_signature),
            "issue": "MODEL_OF composition mismatch",
        })
    return rows


def metric_semantic_issues(graph: nx.Graph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        if str(data.get("type", "")) != "Measurement":
            continue
        current = str(data.get("metric_id", ""))
        expected = refine_semantic_metric_id(
            entry_id=current,
            label=str(data.get("metric") or data.get("label") or ""),
            source_texts=(
                data.get("source_expression"),
                data.get("description"),
                data.get("basis"),
            ),
        )
        if expected == current:
            continue
        rows.append({
            "id": str(node_id),
            "current_metric_id": current,
            "expected_metric_id": expected,
            "label": data.get("label", ""),
            "source_expression": data.get("source_expression", ""),
            "description": data.get("description", ""),
            "basis": data.get("basis", ""),
            "issue": "source expression conflicts with metric category",
        })
    return rows


def _claim_statement(graph: nx.Graph, node_id: str) -> str:
    data = graph.nodes[node_id]
    return str(data.get("statement") or data.get("label") or node_id)


def exact_duplicate_claim_detection_issues(
    graph: nx.Graph,
    candidates: Sequence[ClaimOverlapCandidate],
) -> list[dict[str, Any]]:
    detected = {
        frozenset((candidate.left_id, candidate.right_id))
        for candidate in candidates
        if candidate.suggested_relation
        in {"EXACT_DUPLICATE", "SAME_CONCLUSION_DIFFERENT_EVIDENCE"}
    }
    groups: dict[tuple[str, str], list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        node_type = str(data.get("type", ""))
        if node_type not in {"ObservationClaim", "MechanismClaim"}:
            continue
        normalized = normalize_scientific_text(_claim_statement(graph, str(node_id)))
        if normalized:
            groups.setdefault((node_type, normalized), []).append(str(node_id))

    issues: list[dict[str, Any]] = []
    for (node_type, normalized), node_ids in groups.items():
        for left_id, right_id in itertools.combinations(sorted(node_ids), 2):
            if frozenset((left_id, right_id)) in detected:
                continue
            issues.append({
                "left_id": left_id,
                "right_id": right_id,
                "claim_node_type": node_type,
                "normalized_statement": normalized,
                "left_statement": _claim_statement(graph, left_id),
                "right_statement": _claim_statement(graph, right_id),
                "issue": "exact normalized duplicate was not surfaced by claim audit",
            })
    return issues


def _node_documents(graph: nx.Graph, node_id: str) -> frozenset[str]:
    documents: set[str] = set()
    for _, _, data in graph.in_edges(node_id, data=True):
        document_id = str(data.get("document_id", "")).strip()
        if document_id:
            documents.add(document_id)
    for _, _, data in graph.out_edges(node_id, data=True):
        document_id = str(data.get("document_id", "")).strip()
        if document_id:
            documents.add(document_id)
    return frozenset(documents)


def cross_document_duplicate_issues(graph: nx.Graph) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        node_type = str(data.get("type", ""))
        if node_type not in _CROSS_DOCUMENT_TYPES:
            continue
        normalized = normalize_scientific_text(data.get("label", ""))
        if normalized:
            groups.setdefault((node_type, normalized), []).append(str(node_id))

    rows: list[dict[str, Any]] = []
    for (node_type, normalized), node_ids in groups.items():
        if len(node_ids) < 2:
            continue
        for left_id, right_id in itertools.combinations(sorted(node_ids), 2):
            left_documents = _node_documents(graph, left_id)
            right_documents = _node_documents(graph, right_id)
            if not left_documents or not right_documents:
                continue
            if not (left_documents - right_documents or right_documents - left_documents):
                continue
            left_signature = node_composition_signature(graph, left_id)
            right_signature = node_composition_signature(graph, right_id)
            if left_signature and right_signature and left_signature != right_signature:
                continue
            rows.append({
                "left_id": left_id,
                "right_id": right_id,
                "type": node_type,
                "normalized_label": normalized,
                "left_documents": sorted(left_documents),
                "right_documents": sorted(right_documents),
                "left_composition": dict(left_signature),
                "right_composition": dict(right_signature),
                "confidence": "high",
                "issue": "unresolved exact-label entity duplicate across documents",
            })
    return rows


def si_figure_provenance_issues(
    graph: nx.Graph,
    *,
    manifest_assets: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report missing pixel provenance only for visual SI locators.

    Supplementary tables are normally represented as Markdown/table blocks and
    therefore do not require a JPEG/PNG pointer. Figure and scheme locators do.
    """
    assets_by_document_page: dict[tuple[str, int], list[str]] = {}
    si_documents_with_assets: set[str] = set()
    for asset in manifest_assets:
        if str(asset.get("document_role", "")) != "supporting_information":
            continue
        document_id = str(asset.get("document_id", ""))
        si_documents_with_assets.add(document_id)
        page_id = asset.get("page_id")
        if page_id is not None:
            try:
                assets_by_document_page.setdefault((document_id, int(page_id)), []).append(
                    str(asset.get("asset_id", ""))
                )
            except (TypeError, ValueError):
                pass

    rows: list[dict[str, Any]] = []
    for source, target, key, data in _edge_iter(graph):
        if str(data.get("document_role", "")) != "supporting_information":
            continue
        document_id = str(data.get("document_id", ""))
        if document_id not in si_documents_with_assets:
            continue
        pointers = _json_list(data.get("evidence_pointers_json"))
        for pointer_index, pointer in enumerate(pointers):
            if not isinstance(pointer, dict):
                continue
            locator_source = " | ".join(
                str(value or "")
                for value in (
                    pointer.get("locator_text"),
                    data.get("section"),
                    data.get("subsection"),
                )
            )
            # A Markdown table block is valid provenance even without pixels.
            if _TABLE_LOCATOR_RE.search(locator_source):
                continue
            if not _VISUAL_LOCATOR_RE.search(locator_source):
                continue
            if pointer.get("asset_ids"):
                continue
            page_id = pointer.get("page_id")
            page_assets: list[str] = []
            try:
                if page_id is not None:
                    page_assets = assets_by_document_page.get((document_id, int(page_id)), [])
            except (TypeError, ValueError):
                pass
            rows.append({
                "source": str(source),
                "relation": str(data.get("relation", "")),
                "target": str(target),
                "edge_key": str(key),
                "pointer_index": pointer_index,
                "document_id": document_id,
                "page_id": page_id,
                "locator_text": pointer.get("locator_text"),
                "locator_key": pointer.get("locator_key"),
                "locator_mapping_method": pointer.get("locator_mapping_method"),
                "available_page_asset_ids": page_assets,
                "issue": "SI figure/scheme locator has no asset pointer",
            })
    return rows

# dac_her/semantic_audit.py
def measurement_subject_consistency(
    graph: nx.Graph,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    issues: list[dict[str, Any]] = []
    multi_provenance: list[dict[str, Any]] = []

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "Measurement":
            continue

        measurement_id = str(node_id)
        expected_target = str(attrs.get("subject_id", ""))

        raw_targets = [
            str(target)
            for _, target, edge_data in graph.out_edges(
                measurement_id,
                data=True,
            )
            if edge_data.get("relation") == "MEASURED_FOR"
        ]

        unique_targets = sorted(set(raw_targets))

        if unique_targets != [expected_target]:
            issues.append({
                "id": measurement_id,
                "subject_id": expected_target,
                "measured_for_targets": raw_targets,
                "unique_targets": unique_targets,
                "issue": (
                    "MEASURED_FOR semantic targets do not "
                    "match Measurement.subject_id"
                ),
            })
            continue

        if len(raw_targets) > 1:
            multi_provenance.append({
                "id": measurement_id,
                "subject_id": expected_target,
                "parallel_edge_count": len(raw_targets),
                "target": expected_target,
                "issue": (
                    "multiple provenance-bearing MEASURED_FOR "
                    "edges share one semantic target"
                ),
            })

    return issues, multi_provenance


def semantic_readiness(
    *,
    model_of_issues: Sequence[dict[str, Any]],
    metric_issues: Sequence[dict[str, Any]],
    exact_claim_detection_issues: Sequence[dict[str, Any]],
    cross_document_duplicates: Sequence[dict[str, Any]],
    si_figure_issues: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "composition_incompatible_model_of": len(model_of_issues),
        "source_metric_mismatches": len(metric_issues),
        "undetected_exact_duplicate_claims": len(exact_claim_detection_issues),
        "unresolved_high_confidence_cross_document_duplicates": len(cross_document_duplicates),
        "si_figure_edges_missing_asset_pointer": len(si_figure_issues),
    }
    payload["passes_semantic_gate"] = all(value == 0 for value in payload.values())
    return payload
