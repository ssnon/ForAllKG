from __future__ import annotations

import networkx as nx

from pipeline_core.discovery.higher_order_operationalization_bridge_candidates import (
    resolve_target_facet_bridge_candidates,
)
from pipeline_core.discovery.higher_order_operationalization_target_facets import (
    OperationalizationTargetFacet,
    TargetFacetRequirementResolution,
    TargetFacetResolution,
    TargetFacetResolutionSet,
)


def _facet_resolution(facet_text, qualifier, head, term):
    facet = OperationalizationTargetFacet(
        facet_id="facet:1",
        original_target="electromagnetic hotspot location and intensity",
        facet_text=facet_text,
        head_text=head,
        qualifier_text=qualifier,
        facet_term=term,
        decomposition_mode="coordinated_head_facet",
    )
    row = TargetFacetResolution(
        requirement_id="req:1",
        output_experiment_id="exp:1",
        original_target=facet.original_target,
        facet=facet,
        status="PARTIAL_METRIC_IDENTITY_SUPPORT",
        retrieval_candidate_count=0,
        exact_metric_identity_candidate_count=0,
        partial_metric_identity_candidate_count=0,
        best_metric_identity_coverage=0.0,
        candidates=[],
    )
    requirement = TargetFacetRequirementResolution(
        requirement_id="req:1",
        output_experiment_id="exp:1",
        original_target=facet.original_target,
        decomposition_mode="coordinated_head_facet",
        facet_count=1,
        exact_supported_facet_count=0,
        partial_supported_facet_count=1,
        unsupported_facet_count=0,
        all_facets_exactly_supported=False,
        facets=[row],
    )
    return TargetFacetResolutionSet(
        requirement_count=1,
        resolution_count=1,
        total_facet_count=1,
        exact_supported_facet_count=0,
        partial_supported_facet_count=1,
        unsupported_facet_count=0,
        status_counts={"PARTIAL_METRIC_IDENTITY_SUPPORT": 1},
        resolutions=[requirement],
    )


def _add_measurement(
    graph,
    *,
    mid,
    paper,
    metric_id,
    metric,
    source_expression,
):
    provider = "calc:" + mid
    graph.add_node(
        provider,
        type="Calculation",
        source_paper_id=paper,
    )
    graph.add_node(
        mid,
        type="Measurement",
        metric_id=metric_id,
        metric=metric,
        label=metric,
        source_expression=source_expression,
        subject_id="subject:" + paper,
        source_paper_id=paper,
    )
    graph.add_edge(
        provider,
        mid,
        relation="HAS_MEASUREMENT",
        paper_id=paper,
    )


def test_repeated_intensity_bridge_candidate_uses_same_measurement_evidence():
    graph = nx.MultiDiGraph()
    for idx, paper in enumerate(("paper:1", "paper:2"), start=1):
        _add_measurement(
            graph,
            mid=f"m:{idx}",
            paper=paper,
            metric_id="local_field_enhancement",
            metric="Local electromagnetic field enhancement",
            source_expression="The hot-spot intensity increased strongly.",
        )

    result = resolve_target_facet_bridge_candidates(
        graph=graph,
        facet_resolutions=_facet_resolution(
            "electromagnetic hotspot intensity",
            "electromagnetic",
            "hotspot",
            "intensity",
        ),
    )

    row = result.resolutions[0]
    assert row.status == "CORPUS_REPEATED_BRIDGE_CANDIDATE"
    assert row.repeated_candidate_count == 1
    candidate = row.candidates[0]
    assert candidate.metric_id == "local_field_enhancement"
    assert candidate.support_paper_count == 2
    assert candidate.scientific_equivalence_asserted is False
    assert candidate.operationalization_bridge_verified is False


def test_metric_qualifier_without_source_head_facet_is_not_bridge():
    graph = nx.MultiDiGraph()
    _add_measurement(
        graph,
        mid="m:1",
        paper="paper:1",
        metric_id="local_field_enhancement",
        metric="Local electromagnetic field enhancement",
        source_expression="The field enhancement increased.",
    )

    result = resolve_target_facet_bridge_candidates(
        graph=graph,
        facet_resolutions=_facet_resolution(
            "electromagnetic hotspot intensity",
            "electromagnetic",
            "hotspot",
            "intensity",
        ),
    )

    assert result.resolutions[0].status == "NO_BRIDGE_CANDIDATE"


def test_location_is_not_inferred_from_lateral_dimension_without_location_text():
    graph = nx.MultiDiGraph()
    _add_measurement(
        graph,
        mid="m:1",
        paper="paper:1",
        metric_id="unregistered_electromagnetic_hotspot_lateral_dimension",
        metric="Electromagnetic hotspot lateral dimension",
        source_expression="The hotspots averaged 14 nm in lateral extent.",
    )

    result = resolve_target_facet_bridge_candidates(
        graph=graph,
        facet_resolutions=_facet_resolution(
            "electromagnetic hotspot location",
            "electromagnetic",
            "hotspot",
            "location",
        ),
    )

    assert result.resolutions[0].status == "NO_BRIDGE_CANDIDATE"


def test_single_paper_support_stays_single_paper_candidate():
    graph = nx.MultiDiGraph()
    _add_measurement(
        graph,
        mid="m:1",
        paper="paper:1",
        metric_id="local_field_enhancement",
        metric="Local electromagnetic field enhancement",
        source_expression="The hot spot intensity increased.",
    )

    result = resolve_target_facet_bridge_candidates(
        graph=graph,
        facet_resolutions=_facet_resolution(
            "electromagnetic hotspot intensity",
            "electromagnetic",
            "hotspot",
            "intensity",
        ),
    )

    row = result.resolutions[0]
    assert row.status == "SINGLE_PAPER_BRIDGE_CANDIDATE"
    assert row.single_paper_candidate_count == 1


def test_bridge_candidates_never_create_authority():
    graph = nx.MultiDiGraph()

    result = resolve_target_facet_bridge_candidates(
        graph=graph,
        facet_resolutions=_facet_resolution(
            "electromagnetic hotspot intensity",
            "electromagnetic",
            "hotspot",
            "intensity",
        ),
    )

    assert result.metric_identity_uses_source_expression is False
    assert result.source_expression_used_as_bridge_evidence_only is True
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

def test_plural_space_separated_hot_spots_is_orthographic_bridge_evidence():
    graph = nx.MultiDiGraph()
    _add_measurement(
        graph,
        mid="m:plural",
        paper="paper:1",
        metric_id="local_field_enhancement",
        metric="Local electromagnetic field enhancement",
        source_expression="The hot spots intensity increased.",
    )

    result = resolve_target_facet_bridge_candidates(
        graph=graph,
        facet_resolutions=_facet_resolution(
            "electromagnetic hotspot intensity",
            "electromagnetic",
            "hotspot",
            "intensity",
        ),
    )

    row = result.resolutions[0]
    assert row.status == "SINGLE_PAPER_BRIDGE_CANDIDATE"
    assert row.candidate_count == 1
    assert row.candidates[0].metric_id == "local_field_enhancement"

def test_required_semantic_terms_do_not_gain_adjacent_phrase_compound():
    from pipeline_core.discovery.higher_order_operationalization_bridge_candidates import (
        _required_terms,
        _tokens,
    )

    required = _required_terms("hotspot intensity")
    searchable = _tokens("hotspot intensity")

    assert required == {"hotspot", "intensity"}
    assert "hotspotintensity" in searchable
    assert "hotspotintensity" not in required
