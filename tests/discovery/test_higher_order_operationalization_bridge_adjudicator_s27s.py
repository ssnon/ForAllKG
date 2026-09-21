from __future__ import annotations

import networkx as nx

from pipeline_core.discovery.higher_order_operationalization_bridge_adjudicator import (
    adjudicate_operationalization_bridges,
)
from pipeline_core.discovery.higher_order_operationalization_bridge_candidates import (
    OperationalizationBridgeCandidate,
    OperationalizationBridgeSupport,
    TargetFacetBridgeResolution,
    TargetFacetBridgeResolutionSet,
)


def _support(mid="m:1", provider_types=None):
    return OperationalizationBridgeSupport(
        measurement_node_id=mid,
        metric_id="local_field_enhancement",
        metric_label="Local electromagnetic field enhancement",
        source_expression="The hot spot intensity increased.",
        source_paper_ids=["paper:1"],
        subject_id="sample:1",
        provider_types=provider_types or ["Calculation"],
        metric_identity_matched_tokens=["electromagnetic"],
        source_expression_matched_tokens=["hotspot", "intensity"],
        metric_supports_qualifier=True,
        source_expression_supports_head_and_facet=True,
    )


def _candidate(status="CORPUS_REPEATED_BRIDGE_CANDIDATE"):
    support = _support()
    return OperationalizationBridgeCandidate(
        bridge_candidate_id="bridge:1",
        requirement_id="req:1",
        output_experiment_id="exp:1",
        original_target="electromagnetic hotspot location and intensity",
        facet_id="facet:intensity",
        facet_text="electromagnetic hotspot intensity",
        qualifier_text="electromagnetic",
        head_text="hotspot",
        facet_term="intensity",
        metric_id="local_field_enhancement",
        metric_labels=["Local electromagnetic field enhancement"],
        support_measurement_count=1,
        support_paper_count=(2 if status.startswith("CORPUS") else 1),
        support_subject_count=1,
        support_provider_types=["Calculation"],
        status=status,
        supports=[support],
    )


def _bridges(candidate=None):
    candidates = [] if candidate is None else [candidate]
    status = (
        "NO_BRIDGE_CANDIDATE"
        if candidate is None
        else candidate.status
    )
    resolution = TargetFacetBridgeResolution(
        requirement_id="req:1",
        output_experiment_id="exp:1",
        facet_id="facet:intensity",
        facet_text="electromagnetic hotspot intensity",
        status=status,
        candidate_count=len(candidates),
        repeated_candidate_count=int(
            candidate is not None
            and candidate.status == "CORPUS_REPEATED_BRIDGE_CANDIDATE"
        ),
        single_paper_candidate_count=int(
            candidate is not None
            and candidate.status == "SINGLE_PAPER_BRIDGE_CANDIDATE"
        ),
        candidates=candidates,
    )
    return TargetFacetBridgeResolutionSet(
        requirement_count=1,
        facet_count=1,
        resolution_count=1,
        candidate_count=len(candidates),
        repeated_candidate_count=resolution.repeated_candidate_count,
        single_paper_candidate_count=(
            resolution.single_paper_candidate_count
        ),
        unresolved_facet_count=int(candidate is None),
        status_counts={status: 1},
        resolutions=[resolution],
    )


def _graph(locator=True, provider_type="Calculation"):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "calc:1",
        type=provider_type,
        source_paper_id="paper:1",
    )
    graph.add_node(
        "m:1",
        type="Measurement",
        metric_id="local_field_enhancement",
    )
    attrs = {
        "relation": "HAS_MEASUREMENT",
        "paper_id": "paper:1",
    }
    if locator:
        attrs.update(
            {
                "chunk_id": "chunk:1",
                "document_id": "main",
                "evidence_pointers_json": "[]",
            }
        )
    graph.add_edge("calc:1", "m:1", **attrs)
    return graph


def test_repeated_provenance_located_candidate_routes_to_source_review():
    result = adjudicate_operationalization_bridges(
        graph=_graph(locator=True),
        bridges=_bridges(_candidate()),
    )

    row = result.facets[0].candidates[0]
    assert row.all_supports_provenance_located is True
    assert (
        "PROVENANCE_LOCATED_SOURCE_REVIEW_REQUIRED"
        in row.review_routes
    )
    assert (
        "CALCULATION_ONLY_BRIDGE_REVIEW_REQUIRED"
        in row.review_routes
    )


def test_missing_locator_routes_to_provenance_repair():
    result = adjudicate_operationalization_bridges(
        graph=_graph(locator=False),
        bridges=_bridges(_candidate()),
    )

    row = result.facets[0].candidates[0]
    assert row.all_supports_provenance_located is False
    assert "PROVENANCE_REPAIR_REQUIRED" in row.review_routes


def test_single_paper_candidate_requires_more_corpus_support():
    result = adjudicate_operationalization_bridges(
        graph=_graph(locator=True),
        bridges=_bridges(
            _candidate("SINGLE_PAPER_BRIDGE_CANDIDATE")
        ),
    )

    assert (
        "ADDITIONAL_CORPUS_SUPPORT_REQUIRED"
        in result.facets[0].review_routes
    )


def test_no_bridge_routes_to_metric_or_extraction_gap():
    result = adjudicate_operationalization_bridges(
        graph=nx.MultiDiGraph(),
        bridges=_bridges(None),
    )

    assert result.unresolved_facet_count == 1
    assert result.facets[0].review_routes == [
        "METRIC_VOCABULARY_OR_EXTRACTION_GAP"
    ]


def test_adjudication_never_creates_authority():
    result = adjudicate_operationalization_bridges(
        graph=_graph(locator=True),
        bridges=_bridges(_candidate()),
    )

    assert result.source_level_review_completed_count == 0
    assert result.scientific_equivalence_asserted is False
    assert result.operationalization_bridge_verified_count == 0
    assert result.operationalization_bridge_authority is False
    assert result.measurement_independence_verified_count == 0
    assert result.experiment_selection_performed is False
    assert result.rejection_authority is False
    assert result.production_selection_authority is False
    assert result.positive_premise_authority is False
    assert result.gap_authority is False
    assert result.novelty_authority is False

def test_projection_native_provenance_locator_is_accepted():
    graph = nx.MultiDiGraph()
    graph.add_node("calc:1", type="Calculation")
    graph.add_node(
        "m:1",
        type="Measurement",
        metric_id="local_field_enhancement",
    )
    graph.add_edge(
        "calc:1",
        "m:1",
        relation="HAS_MEASUREMENT",
        source_paper_id="paper:1",
        source_paper_ids_json='["paper:1"]',
        source_edge_ids_json='["paper::paper:1::edge:abc"]',
        projection_edge_ids_json='["paper::paper:1::projection:def"]',
        evidence_pointers_json=(
            '[{"document_id":"main","document_role":"main",'
            '"page_id":4,"asset_ids":["asset:1"],'
            '"locator_text":"Figure 3a","locator_key":"figure:3"}]'
        ),
    )

    result = adjudicate_operationalization_bridges(
        graph=graph,
        bridges=_bridges(_candidate()),
    )

    row = result.facets[0].candidates[0]
    assert row.all_supports_provenance_located is True
    assert row.provider_provenance_support_count == 1
    assert (
        "PROVENANCE_LOCATED_SOURCE_REVIEW_REQUIRED"
        in row.review_routes
    )
    assert "PROVENANCE_REPAIR_REQUIRED" not in row.review_routes


def test_projection_source_edge_trace_without_pointer_stays_unlocated():
    graph = nx.MultiDiGraph()
    graph.add_node("calc:1", type="Calculation")
    graph.add_node(
        "m:1",
        type="Measurement",
        metric_id="local_field_enhancement",
    )
    graph.add_edge(
        "calc:1",
        "m:1",
        relation="HAS_MEASUREMENT",
        source_paper_id="paper:1",
        source_paper_ids_json='["paper:1"]',
        source_edge_ids_json='["paper::paper:1::edge:abc"]',
        projection_edge_ids_json='["paper::paper:1::projection:def"]',
        evidence_pointers_json="[]",
    )

    result = adjudicate_operationalization_bridges(
        graph=graph,
        bridges=_bridges(_candidate()),
    )

    row = result.facets[0].candidates[0]
    assert row.all_supports_provenance_located is False
    assert "PROVENANCE_REPAIR_REQUIRED" in row.review_routes
