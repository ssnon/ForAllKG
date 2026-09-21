from __future__ import annotations

import json
from collections import Counter
from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_operationalization_bridge_candidates import (
    OperationalizationBridgeCandidate,
    TargetFacetBridgeResolutionSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


BridgeReviewRoute = Literal[
    "PROVENANCE_LOCATED_SOURCE_REVIEW_REQUIRED",
    "PROVENANCE_REPAIR_REQUIRED",
    "ADDITIONAL_CORPUS_SUPPORT_REQUIRED",
    "METRIC_VOCABULARY_OR_EXTRACTION_GAP",
    "CALCULATION_ONLY_BRIDGE_REVIEW_REQUIRED",
]


class BridgeSupportProvenance(StrictModel):
    measurement_node_id: str
    source_paper_ids: list[str] = Field(default_factory=list)

    incoming_has_measurement_edge_count: int = Field(ge=0)
    outgoing_measured_for_edge_count: int = Field(ge=0)

    provider_edge_locator_count: int = Field(ge=0)
    measured_for_edge_locator_count: int = Field(ge=0)

    provider_edge_provenance_present: bool
    measured_for_edge_provenance_present: bool
    provenance_locator_present: bool

    provider_types: list[str] = Field(default_factory=list)
    calculation_only_provider_context: bool

    provenance_is_scientific_bridge_verification: Literal[False] = False
    operationalization_bridge_verified: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class BridgeCandidateAdjudication(StrictModel):
    bridge_candidate_id: str
    requirement_id: str
    output_experiment_id: str
    facet_id: str
    facet_text: str
    metric_id: str

    original_status: str
    support_measurement_count: int = Field(ge=1)
    support_paper_count: int = Field(ge=1)

    provenance_located_support_count: int = Field(ge=0)
    provider_provenance_support_count: int = Field(ge=0)
    measured_for_provenance_support_count: int = Field(ge=0)

    all_supports_provenance_located: bool
    calculation_only_candidate: bool

    review_routes: list[BridgeReviewRoute] = Field(default_factory=list)
    supports: list[BridgeSupportProvenance] = Field(default_factory=list)

    source_level_review_completed: Literal[False] = False
    scientific_equivalence_asserted: Literal[False] = False
    operationalization_bridge_verified: Literal[False] = False
    operationalization_bridge_authority: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class FacetBridgeAdjudication(StrictModel):
    requirement_id: str
    output_experiment_id: str
    facet_id: str
    facet_text: str

    original_status: str
    candidate_count: int = Field(ge=0)
    adjudicated_candidate_count: int = Field(ge=0)

    review_routes: list[BridgeReviewRoute] = Field(default_factory=list)
    candidates: list[BridgeCandidateAdjudication] = Field(
        default_factory=list
    )

    source_level_review_completed: Literal[False] = False
    operationalization_bridge_verified: Literal[False] = False
    operationalization_bridge_authority: Literal[False] = False
    measurement_independence_verified: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class BridgeAdjudicationSet(StrictModel):
    schema_version: Literal[
        "operationalization-bridge-adjudication-v1"
    ] = "operationalization-bridge-adjudication-v1"

    facet_resolution_count: int = Field(ge=0)
    bridge_candidate_count: int = Field(ge=0)
    provenance_located_candidate_count: int = Field(ge=0)
    calculation_only_candidate_count: int = Field(ge=0)
    unresolved_facet_count: int = Field(ge=0)

    route_counts: dict[str, int] = Field(default_factory=dict)
    facets: list[FacetBridgeAdjudication] = Field(default_factory=list)

    source_level_review_completed_count: Literal[0] = 0
    scientific_equivalence_asserted: Literal[False] = False
    operationalization_bridge_verified_count: Literal[0] = 0
    operationalization_bridge_authority: Literal[False] = False
    measurement_independence_verified_count: Literal[0] = 0
    scientific_quality_ranking_performed: Literal[False] = False
    experiment_selection_performed: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    shadow_only: Literal[True] = True


def _relation(attrs: dict) -> str:
    return str(
        attrs.get("relation")
        or attrs.get("title")
        or ""
    ).strip().upper()


def _json_nonempty(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    return bool(parsed)


def _json_list(value: object) -> list[object]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _projection_pointer_locator_present(value: object) -> bool:
    """
    Accept only explicit projection-carried source locators.

    GraphAgents evidence projections intentionally replace canonical
    top-level chunk/document attributes with provenance payloads on the
    projection edge. A pointer counts as located only when it identifies a
    document and also provides a page, locator string/key, or asset.
    """
    for pointer in _json_list(value):
        if not isinstance(pointer, dict):
            continue

        document_id = str(pointer.get("document_id") or "").strip()
        page_id = pointer.get("page_id")
        locator_text = str(pointer.get("locator_text") or "").strip()
        locator_key = str(pointer.get("locator_key") or "").strip()

        asset_ids = pointer.get("asset_ids")
        has_assets = bool(
            isinstance(asset_ids, list)
            and any(str(item).strip() for item in asset_ids)
        )

        has_local_locator = bool(
            page_id not in (None, "")
            or locator_text
            or locator_key
            or has_assets
        )

        if document_id and has_local_locator:
            return True

    return False


def _edge_locator_present(attrs: dict) -> bool:
    """
    Recognize both canonical-edge and projection-native provenance schemas.

    Canonical edges can expose paper/chunk/document fields directly.
    GraphAgents evidence projection edges instead preserve the canonical
    lineage through source_paper_ids_json/source_edge_ids_json plus
    evidence_pointers_json. Neither representation implies scientific
    verification; this predicate only establishes that source review can
    be traced to a concrete document locator.
    """
    paper_id = str(
        attrs.get("paper_id")
        or attrs.get("source_paper_id")
        or ""
    ).strip()
    chunk_id = str(attrs.get("chunk_id") or "").strip()
    document_id = str(attrs.get("document_id") or "").strip()

    evidence_pointer = _json_nonempty(
        attrs.get("evidence_pointers_json")
    )
    page_locator = _json_nonempty(attrs.get("page_ids_json"))
    asset_locator = _json_nonempty(
        attrs.get("evidence_asset_ids_json")
    ) or _json_nonempty(attrs.get("asset_ids_json"))

    canonical_locator = bool(
        paper_id
        and chunk_id
        and (
            document_id
            or evidence_pointer
            or page_locator
            or asset_locator
        )
    )

    source_paper_ids = {
        str(value).strip()
        for value in _json_list(
            attrs.get("source_paper_ids_json")
        )
        if str(value).strip()
    }
    if str(attrs.get("source_paper_id") or "").strip():
        source_paper_ids.add(
            str(attrs.get("source_paper_id")).strip()
        )

    source_edge_ids = {
        str(value).strip()
        for value in _json_list(
            attrs.get("source_edge_ids_json")
        )
        if str(value).strip()
    }

    projection_locator = bool(
        source_paper_ids
        and source_edge_ids
        and _projection_pointer_locator_present(
            attrs.get("evidence_pointers_json")
        )
    )

    return bool(canonical_locator or projection_locator)


def _support_provenance(
    graph: nx.Graph,
    *,
    measurement_node_id: str,
    source_paper_ids: list[str],
    provider_types_from_support: list[str],
) -> BridgeSupportProvenance:
    incoming_count = 0
    outgoing_count = 0
    provider_locator_count = 0
    measured_for_locator_count = 0
    provider_types = set(provider_types_from_support)

    if graph.is_multigraph():
        incoming = graph.in_edges(
            measurement_node_id,
            keys=True,
            data=True,
        )
        for source, _, _, attrs in incoming:
            attrs = dict(attrs)
            if _relation(attrs) != "HAS_MEASUREMENT":
                continue
            incoming_count += 1
            if _edge_locator_present(attrs):
                provider_locator_count += 1
            if graph.has_node(source):
                provider_type = str(
                    graph.nodes[source].get("type", "") or ""
                ).strip()
                if provider_type:
                    provider_types.add(provider_type)

        outgoing = graph.out_edges(
            measurement_node_id,
            keys=True,
            data=True,
        )
        for _, _, _, attrs in outgoing:
            attrs = dict(attrs)
            if _relation(attrs) != "MEASURED_FOR":
                continue
            outgoing_count += 1
            if _edge_locator_present(attrs):
                measured_for_locator_count += 1
    else:
        for source, _, attrs in graph.in_edges(
            measurement_node_id,
            data=True,
        ):
            attrs = dict(attrs)
            if _relation(attrs) != "HAS_MEASUREMENT":
                continue
            incoming_count += 1
            if _edge_locator_present(attrs):
                provider_locator_count += 1
            if graph.has_node(source):
                provider_type = str(
                    graph.nodes[source].get("type", "") or ""
                ).strip()
                if provider_type:
                    provider_types.add(provider_type)

        for _, _, attrs in graph.out_edges(
            measurement_node_id,
            data=True,
        ):
            attrs = dict(attrs)
            if _relation(attrs) != "MEASURED_FOR":
                continue
            outgoing_count += 1
            if _edge_locator_present(attrs):
                measured_for_locator_count += 1

    provider_provenance = provider_locator_count > 0
    measured_for_provenance = measured_for_locator_count > 0

    # A support is provenance-located when the measurement-producing relation
    # can be traced to a paper/chunk/document locator. MEASURED_FOR provenance
    # is reported separately rather than required, because older canonical
    # graphs can preserve measurement production provenance without a distinct
    # MEASURED_FOR locator.
    provenance_locator_present = provider_provenance

    provider_type_list = sorted(
        value
        for value in provider_types
        if value
    )
    calculation_only = bool(
        provider_type_list
        and set(provider_type_list) <= {"Calculation"}
    )

    return BridgeSupportProvenance(
        measurement_node_id=measurement_node_id,
        source_paper_ids=list(source_paper_ids),
        incoming_has_measurement_edge_count=incoming_count,
        outgoing_measured_for_edge_count=outgoing_count,
        provider_edge_locator_count=provider_locator_count,
        measured_for_edge_locator_count=measured_for_locator_count,
        provider_edge_provenance_present=provider_provenance,
        measured_for_edge_provenance_present=measured_for_provenance,
        provenance_locator_present=provenance_locator_present,
        provider_types=provider_type_list,
        calculation_only_provider_context=calculation_only,
    )


def _candidate_adjudication(
    graph: nx.Graph,
    candidate: OperationalizationBridgeCandidate,
) -> BridgeCandidateAdjudication:
    support_rows = [
        _support_provenance(
            graph,
            measurement_node_id=support.measurement_node_id,
            source_paper_ids=support.source_paper_ids,
            provider_types_from_support=support.provider_types,
        )
        for support in candidate.supports
    ]

    provenance_count = sum(
        row.provenance_locator_present
        for row in support_rows
    )
    provider_provenance_count = sum(
        row.provider_edge_provenance_present
        for row in support_rows
    )
    measured_for_provenance_count = sum(
        row.measured_for_edge_provenance_present
        for row in support_rows
    )

    all_located = bool(
        support_rows
        and provenance_count == len(support_rows)
    )
    calculation_only = bool(
        support_rows
        and all(
            row.calculation_only_provider_context
            for row in support_rows
        )
    )

    routes: list[BridgeReviewRoute] = []

    if candidate.status == "CORPUS_REPEATED_BRIDGE_CANDIDATE":
        if all_located:
            routes.append(
                "PROVENANCE_LOCATED_SOURCE_REVIEW_REQUIRED"
            )
        else:
            routes.append("PROVENANCE_REPAIR_REQUIRED")
    else:
        routes.append("ADDITIONAL_CORPUS_SUPPORT_REQUIRED")

    if calculation_only:
        routes.append(
            "CALCULATION_ONLY_BRIDGE_REVIEW_REQUIRED"
        )

    return BridgeCandidateAdjudication(
        bridge_candidate_id=candidate.bridge_candidate_id,
        requirement_id=candidate.requirement_id,
        output_experiment_id=candidate.output_experiment_id,
        facet_id=candidate.facet_id,
        facet_text=candidate.facet_text,
        metric_id=candidate.metric_id,
        original_status=candidate.status,
        support_measurement_count=candidate.support_measurement_count,
        support_paper_count=candidate.support_paper_count,
        provenance_located_support_count=provenance_count,
        provider_provenance_support_count=provider_provenance_count,
        measured_for_provenance_support_count=(
            measured_for_provenance_count
        ),
        all_supports_provenance_located=all_located,
        calculation_only_candidate=calculation_only,
        review_routes=routes,
        supports=support_rows,
    )


def adjudicate_operationalization_bridges(
    *,
    graph: nx.Graph,
    bridges: TargetFacetBridgeResolutionSet,
) -> BridgeAdjudicationSet:
    facet_rows = []

    for resolution in bridges.resolutions:
        if not resolution.candidates:
            facet_rows.append(
                FacetBridgeAdjudication(
                    requirement_id=resolution.requirement_id,
                    output_experiment_id=(
                        resolution.output_experiment_id
                    ),
                    facet_id=resolution.facet_id,
                    facet_text=resolution.facet_text,
                    original_status=resolution.status,
                    candidate_count=0,
                    adjudicated_candidate_count=0,
                    review_routes=[
                        "METRIC_VOCABULARY_OR_EXTRACTION_GAP"
                    ],
                    candidates=[],
                )
            )
            continue

        candidates = [
            _candidate_adjudication(graph, candidate)
            for candidate in resolution.candidates
        ]

        routes: list[BridgeReviewRoute] = []
        for candidate in candidates:
            for route in candidate.review_routes:
                if route not in routes:
                    routes.append(route)

        facet_rows.append(
            FacetBridgeAdjudication(
                requirement_id=resolution.requirement_id,
                output_experiment_id=resolution.output_experiment_id,
                facet_id=resolution.facet_id,
                facet_text=resolution.facet_text,
                original_status=resolution.status,
                candidate_count=resolution.candidate_count,
                adjudicated_candidate_count=len(candidates),
                review_routes=routes,
                candidates=candidates,
            )
        )

    route_counts = Counter(
        route
        for facet in facet_rows
        for route in facet.review_routes
    )

    all_candidates = [
        candidate
        for facet in facet_rows
        for candidate in facet.candidates
    ]

    return BridgeAdjudicationSet(
        facet_resolution_count=len(facet_rows),
        bridge_candidate_count=len(all_candidates),
        provenance_located_candidate_count=sum(
            candidate.all_supports_provenance_located
            for candidate in all_candidates
        ),
        calculation_only_candidate_count=sum(
            candidate.calculation_only_candidate
            for candidate in all_candidates
        ),
        unresolved_facet_count=sum(
            "METRIC_VOCABULARY_OR_EXTRACTION_GAP"
            in facet.review_routes
            for facet in facet_rows
        ),
        route_counts=dict(sorted(route_counts.items())),
        facets=facet_rows,
    )
