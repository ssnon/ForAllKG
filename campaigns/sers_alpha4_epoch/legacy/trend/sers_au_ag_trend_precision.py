from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping

import networkx as nx

from campaigns.sers_alpha4_epoch.legacy.trend.sers_au_ag_trend_alpha4c21 import (
    SERS_AU_AG_TREND_SEMANTICS_ID,
    SERS_TREND_CONTROL_SPECS,
    extract_control_landmark_from_text,
)
from dac_her.trend_precision import (
    PaperLocalTrendResult,
    TrendEvidenceAnnotation,
    TrendPrecisionAdapter,
    stable_local_trend_result_id,
)


SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID = (
    "sers_au_ag_trend_precision_v1_alpha4c21"
)

_CALCULATION_TOKENS = re.compile(
    r"\b(?:dda|discrete\s+dipole|simulation|simulated|simulation-based|"
    r"calculation|calculated|computed|finite[-\s]+element|theoretical|"
    r"model(?:ing|ling)?)\b",
    re.I,
)

_COMMON_SUBJECT_TOKENS = frozenset({
    "substrate", "sers", "sample", "material", "nanoparticle",
})


def _norm(value: Any) -> str:
    text = str(value or "").casefold()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-").replace("@", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _singularize(token: str) -> str:
    replacements = {
        "nanoboxes": "nanobox",
        "nanocubes": "nanocube",
        "nanostars": "nanostar",
        "nanoplates": "nanoplate",
        "nanoparticles": "nanoparticle",
        "particles": "particle",
        "substrates": "substrate",
        "cubes": "cube",
        "boxes": "box",
    }
    return replacements.get(token, token)


def _normalize_subject_token(token: str) -> str:
    token = _singularize(token)
    token = re.sub(r"\d.*$", "", token)
    return token


def _subject_family_signature(
    graph: nx.Graph,
    subject_ids: tuple[str, ...],
) -> str:
    signatures: list[str] = []
    for node_id in subject_ids:
        parts = [str(node_id)]
        if node_id in graph:
            attrs = graph.nodes[node_id]
            for key in ("label", "name", "description"):
                value = str(attrs.get(key, "")).strip()
                if value:
                    parts.append(value)
        tokens: list[str] = []
        for raw in _norm(" ".join(parts)).split():
            token = _normalize_subject_token(raw)
            if not token or token.isdigit() or token in _COMMON_SUBJECT_TOKENS:
                continue
            tokens.append(token)
        if tokens:
            signatures.append(" ".join(sorted(set(tokens))))
    if not signatures:
        return ""
    # Choose the shortest normalized family signature so variable-specific
    # source IDs do not dominate a shared morphology/material label.
    return sorted(set(signatures), key=lambda value: (len(value), value))[0]


def _outgoing_measured_for(
    graph: nx.Graph,
    measurement_id: str,
) -> set[str]:
    subjects: set[str] = set()
    if measurement_id not in graph or not graph.is_directed():
        return subjects
    if graph.is_multigraph():
        iterator = graph.out_edges(measurement_id, keys=True, data=True)
        for _left, right, _key, attrs in iterator:
            if str(attrs.get("relation", "")) == "MEASURED_FOR":
                subjects.add(str(right))
    else:
        iterator = graph.out_edges(measurement_id, data=True)
        for _left, right, attrs in iterator:
            if str(attrs.get("relation", "")) == "MEASURED_FOR":
                subjects.add(str(right))
    return subjects


def _calculation_lexical_support(
    row: Mapping[str, Any],
    graph: nx.Graph,
) -> bool:
    pieces = [
        str(row.get("source_expression", "")),
        *[str(value) for value in row.get("source_expressions", []) or []],
    ]
    for node_id in row.get("source_node_ids", []) or []:
        node_id = str(node_id)
        pieces.append(node_id)
        if node_id in graph:
            attrs = graph.nodes[node_id]
            for key in (
                "label", "name", "description", "statement",
                "source_expression", "method", "node_text",
            ):
                value = str(attrs.get(key, "")).strip()
                if value:
                    pieces.append(value)
    return bool(_CALCULATION_TOKENS.search(" | ".join(pieces)))


def _observable_semantics(key: str) -> str:
    return {
        "raman_intensity": "measured_signal_intensity",
        "sers_enhancement_factor": "formal_sers_enhancement_factor",
        "relative_sers_intensity_ratio": "relative_sers_intensity_ratio",
        "sers_performance": "qualitative_sers_performance",
    }.get(key, f"other:{key}")


def _annotation(
    row: Mapping[str, Any],
    graph: nx.Graph,
) -> TrendEvidenceAnnotation:
    trend_id = str(row.get("trend_id", "")).strip()
    paper_id = str(row.get("paper_id", "")).strip()
    basis = str(row.get("evidence_basis", "")).strip()

    if basis in {"reported_directional_claim", "reported_correlation"}:
        evidence_kind = "reported_claim"
        classification_basis = "claim_contract"
    elif row.get("source_calculation_ids"):
        evidence_kind = "calculated_numeric"
        classification_basis = "explicit_calculation_node"
    elif _calculation_lexical_support(row, graph):
        evidence_kind = "calculated_numeric"
        classification_basis = "explicit_calculation_lineage_text"
    else:
        evidence_kind = "experimental_numeric"
        classification_basis = "measurement_lineage_without_calculation_marker"

    control_key = str(row.get("independent_variable_key", "")).strip()
    spec = SERS_TREND_CONTROL_SPECS.get(control_key)
    control_family = spec.family if spec is not None else "other"

    all_subjects = {
        str(value)
        for value in row.get("subject_ids", []) or []
        if str(value).strip()
    }
    measurement_ids = [
        str(value)
        for value in row.get("source_measurement_ids", []) or []
        if str(value).strip()
    ]
    if basis.startswith("controlled_numeric"):
        trend_subjects: set[str] = set()
        for measurement_id in measurement_ids:
            trend_subjects.update(_outgoing_measured_for(graph, measurement_id))
        if not trend_subjects:
            trend_subjects = set(all_subjects)
        reference_subjects = all_subjects - trend_subjects
    else:
        trend_subjects = set(all_subjects)
        reference_subjects = set()

    landmark = extract_control_landmark_from_text(
        control_key,
        str(row.get("source_expression", "")),
    )

    return TrendEvidenceAnnotation(
        trend_id=trend_id,
        paper_id=paper_id,
        precision_semantics_id=SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
        evidence_kind=evidence_kind,
        classification_basis=classification_basis,
        control_family=control_family,
        observable_semantics=_observable_semantics(
            str(row.get("dependent_observable_key", "")).strip()
        ),
        trend_subject_ids=tuple(sorted(trend_subjects)),
        reference_subject_ids=tuple(sorted(reference_subjects)),
        source_control_value_text=str(landmark.get("source_value_text", "")),
        canonical_control_value_numeric=landmark.get("canonical_value_numeric"),
        canonical_control_unit=str(landmark.get("canonical_unit", "")),
        normalization_transform=str(landmark.get("normalization_transform", "")),
    )


def _make_result(
    *,
    rows: list[Mapping[str, Any]],
    annotation_by_id: dict[str, TrendEvidenceAnnotation],
    result_lane: str,
) -> PaperLocalTrendResult:
    first = rows[0]
    member_ids = tuple(sorted(str(row.get("trend_id", "")) for row in rows))
    member_annotations = [annotation_by_id[trend_id] for trend_id in member_ids]
    result_id = stable_local_trend_result_id(
        paper_id=str(first.get("paper_id", "")),
        result_lane=result_lane,
        independent_variable_key=str(first.get("independent_variable_key", "")),
        dependent_observable_key=str(first.get("dependent_observable_key", "")),
        direction=str(first.get("direction", "")),
        shape=str(first.get("shape", "")),
        member_trend_ids=member_ids,
    )

    def union(field: str) -> tuple[str, ...]:
        return tuple(sorted({
            str(value)
            for row in rows
            for value in row.get(field, []) or []
            if str(value).strip()
        }))

    return PaperLocalTrendResult(
        result_id=result_id,
        paper_id=str(first.get("paper_id", "")),
        domain_profile_id=str(first.get("domain_profile_id", "")),
        trend_semantics_id=str(first.get("trend_semantics_id", "")),
        precision_semantics_id=SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
        result_lane=result_lane,
        independent_variable_key=str(first.get("independent_variable_key", "")),
        dependent_observable_key=str(first.get("dependent_observable_key", "")),
        direction=str(first.get("direction", "")),
        shape=str(first.get("shape", "")),
        control_family=member_annotations[0].control_family,
        observable_semantics=member_annotations[0].observable_semantics,
        member_trend_ids=member_ids,
        evidence_kinds=tuple(sorted({item.evidence_kind for item in member_annotations})),
        trend_subject_ids=tuple(sorted({
            value
            for item in member_annotations
            for value in item.trend_subject_ids
        })),
        reference_subject_ids=tuple(sorted({
            value
            for item in member_annotations
            for value in item.reference_subject_ids
        })),
        source_claim_ids=union("source_claim_ids"),
        source_measurement_ids=union("source_measurement_ids"),
        source_measurement_result_ids=union("source_measurement_result_ids"),
        source_calculation_ids=union("source_calculation_ids"),
        source_node_ids=union("source_node_ids"),
        support_mention_count=len(member_ids),
    )


def _consolidate(
    rows: list[Mapping[str, Any]],
    annotations: list[TrendEvidenceAnnotation],
    graphs: dict[str, nx.Graph],
) -> list[PaperLocalTrendResult]:
    annotation_by_id = {item.trend_id: item for item in annotations}
    numeric_rows: list[Mapping[str, Any]] = []
    claim_groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)

    for row in rows:
        trend_id = str(row.get("trend_id", ""))
        annotation = annotation_by_id[trend_id]
        basis = str(row.get("evidence_basis", ""))
        if basis.startswith("controlled_numeric"):
            numeric_rows.append(row)
            continue

        graph = graphs[str(row.get("paper_id", ""))]
        subject_signature = _subject_family_signature(
            graph,
            annotation.trend_subject_ids,
        )
        key = (
            str(row.get("paper_id", "")),
            str(row.get("independent_variable_key", "")),
            str(row.get("dependent_observable_key", "")),
            str(row.get("direction", "")),
            str(row.get("shape", "")),
            annotation.control_family,
            annotation.observable_semantics,
            subject_signature,
        )
        claim_groups[key].append(row)

    results: list[PaperLocalTrendResult] = []
    for row in numeric_rows:
        results.append(
            _make_result(
                rows=[row],
                annotation_by_id=annotation_by_id,
                result_lane="numeric",
            )
        )
    for _key, group in sorted(claim_groups.items(), key=lambda item: item[0]):
        results.append(
            _make_result(
                rows=group,
                annotation_by_id=annotation_by_id,
                result_lane="claim",
            )
        )
    return sorted(
        results,
        key=lambda row: (
            row.paper_id,
            row.independent_variable_key,
            row.dependent_observable_key,
            row.result_lane,
            row.direction,
            row.shape,
            row.result_id,
        ),
    )


SERS_AU_AG_TREND_PRECISION_ADAPTER = TrendPrecisionAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    precision_semantics_id=SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
    annotate_fn=_annotation,
    consolidate_fn=_consolidate,
)
