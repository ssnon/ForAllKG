from __future__ import annotations

from dataclasses import replace

import networkx as nx

from dac_her.comparison_context import audit_comparison_outputs
from dac_her.comparison_domain import (
    ComparisonAssessment,
    ComparisonContext,
    ComparisonDimensionValue,
    ComparisonDomainAdapter,
)
from dac_her.metric_definition_domain import (
    MetricDefinitionContext,
    MetricDefinitionDomainAdapter,
)
from dac_her.quality_aware_comparison import (
    QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID,
    apply_metric_definition_numeric_gate,
    build_metric_definition_assessments,
)


def _context(
    context_id: str,
    paper_id: str,
    measurement_id: str,
    *,
    observable: str = "sers_enhancement_factor",
) -> ComparisonContext:
    return ComparisonContext(
        context_id=context_id,
        domain_profile_id="sers_au_ag",
        comparison_semantics_id="cmp-v1",
        paper_id=paper_id,
        measurement_id=measurement_id,
        observable_key=observable,
        observable_label=observable,
        value_numeric=1.0,
        value_text="",
        unit="a.u.",
        source_expression="measurement",
        subject_ids=("sub",),
        dimensions=(
            ComparisonDimensionValue(
                name="analyte",
                status="known",
                normalized_value="mb",
                source_values=("MB",),
                source_node_ids=(measurement_id,),
            ),
        ),
        source_node_ids=(measurement_id,),
    )


def _assessment(
    left: ComparisonContext,
    right: ComparisonContext,
    *,
    ranking_mode: str = "allowed_if_complete",
    numeric_allowed: bool = True,
) -> ComparisonAssessment:
    return ComparisonAssessment(
        assessment_id=(
            f"comparison:{left.context_id}:{right.context_id}"
        ),
        comparison_semantics_id="cmp-v1",
        observable_key=left.observable_key,
        observable_policy_id="sers_ef_v1",
        observable_family="sers_performance",
        applicable_dimensions=("analyte",),
        ranking_required_dimensions=("analyte",),
        numeric_ranking_mode=ranking_mode,
        ranking_direction=(
            "higher_better"
            if ranking_mode != "disabled"
            else "none"
        ),
        left_context_id=left.context_id,
        right_context_id=right.context_id,
        left_paper_id=left.paper_id,
        right_paper_id=right.paper_id,
        compatibility="compatible",
        measurement_unit_status="matched",
        matched_dimensions=("analyte",),
        mismatched_dimensions=(),
        unknown_dimensions=(),
        ambiguous_dimensions=(),
        reasons=(),
        numeric_ranking_allowed=numeric_allowed,
    )


def _metric(
    context_id: str,
    paper_id: str,
    measurement_id: str,
    *,
    family: str,
    aggregation: str = "unspecified",
    normalization: str,
    reference: str = "normal_raman",
    criterion: str = "",
) -> MetricDefinitionContext:
    return MetricDefinitionContext(
        context_id=context_id,
        domain_profile_id="sers_au_ag",
        metric_definition_semantics_id="metric-v1",
        paper_id=paper_id,
        measurement_id=measurement_id,
        observable_key="sers_enhancement_factor",
        definition_status="known",
        definition_family=family,
        aggregation_scope=aggregation,
        normalization_basis=normalization,
        reference_basis=reference,
        criterion=criterion,
        source_expression="definition",
        source_measurement_ids=(measurement_id,),
        source_node_ids=(measurement_id,),
    )


def _metric_adapter() -> MetricDefinitionDomainAdapter:
    return MetricDefinitionDomainAdapter(
        adapter_id="sers_au_ag",
        domain_profile_id="sers_au_ag",
        semantics_id="metric-v1",
        supported_observable_keys=frozenset({
            "sers_enhancement_factor",
        }),
        definition_families=frozenset({
            "molecule_normalized_intensity_ratio",
            "concentration_normalized_intensity_ratio",
        }),
        aggregation_scopes=frozenset({
            "single_particle",
            "population_mean",
            "unspecified",
        }),
        normalization_bases=frozenset({
            "molecule_count",
            "concentration",
        }),
        reference_bases=frozenset({
            "normal_raman",
        }),
        extract_contexts_fn=lambda _graph, _paper: [],
    )


def _comparison_adapter() -> ComparisonDomainAdapter:
    return ComparisonDomainAdapter(
        adapter_id="sers_au_ag",
        domain_profile_id="sers_au_ag",
        semantics_id="cmp-v1",
        dimensions=("analyte",),
        required_for_numeric_ranking=frozenset({"analyte"}),
        extract_contexts_fn=lambda _graph, _paper: [],
    )


def _graphs() -> dict[str, nx.MultiDiGraph]:
    graphs: dict[str, nx.MultiDiGraph] = {}
    for paper_id, measurement_id in (("P1", "m1"), ("P2", "m2")):
        graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
        graph.add_node(measurement_id, type="Measurement")
        graphs[paper_id] = graph
    return graphs


def test_alpha4b3b4c1_known_family_mismatch_precedes_unknown_aggregation():
    left = _context("c1", "P1", "m1")
    right = _context("c2", "P2", "m2")
    base = _assessment(left, right)

    rows = build_metric_definition_assessments(
        comparison_assessments=[base],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            _metric(
                "d1",
                "P1",
                "m1",
                family="molecule_normalized_intensity_ratio",
                normalization="molecule_count",
                aggregation="unspecified",
            ),
            _metric(
                "d2",
                "P2",
                "m2",
                family="concentration_normalized_intensity_ratio",
                normalization="concentration",
                aggregation="unspecified",
            ),
        ],
        adapter=_metric_adapter(),
    )

    row = rows[0]
    assert row.compatibility == "different_definition"
    assert row.numeric_metric_definition_gate is False
    assert "metric_definition_family_mismatch" in row.reasons
    assert "metric_normalization_basis_mismatch" in row.reasons
    assert "metric_aggregation_scope_unknown" in row.reasons


def test_alpha4b3b4c1_same_signature_unknown_aggregation_stays_unknown():
    left = _context("c1", "P1", "m1")
    right = _context("c2", "P2", "m2")
    base = _assessment(left, right)

    rows = build_metric_definition_assessments(
        comparison_assessments=[base],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            _metric(
                "d1",
                "P1",
                "m1",
                family="molecule_normalized_intensity_ratio",
                normalization="molecule_count",
                aggregation="unspecified",
            ),
            _metric(
                "d2",
                "P2",
                "m2",
                family="molecule_normalized_intensity_ratio",
                normalization="molecule_count",
                aggregation="unspecified",
            ),
        ],
        adapter=_metric_adapter(),
    )

    row = rows[0]
    assert row.compatibility == "unknown"
    assert row.reasons == ("metric_aggregation_scope_unknown",)


def test_alpha4b3b4c1_ranking_relevance_excludes_disabled_policy():
    left = _context("c1", "P1", "m1")
    right = _context("c2", "P2", "m2")
    disabled = _assessment(
        left,
        right,
        ranking_mode="disabled",
        numeric_allowed=False,
    )

    metric_rows = build_metric_definition_assessments(
        comparison_assessments=[disabled],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            _metric(
                "d1",
                "P1",
                "m1",
                family="molecule_normalized_intensity_ratio",
                normalization="molecule_count",
                aggregation="single_particle",
            ),
            _metric(
                "d2",
                "P2",
                "m2",
                family="molecule_normalized_intensity_ratio",
                normalization="molecule_count",
                aggregation="single_particle",
            ),
        ],
        adapter=_metric_adapter(),
    )
    final = apply_metric_definition_numeric_gate(
        [disabled],
        metric_rows,
    )

    audit = audit_comparison_outputs(
        contexts=[left, right],
        assessments=final,
        source_graphs=_graphs(),
        adapter=_comparison_adapter(),
        metric_definition_assessments=metric_rows,
    )

    assert audit["metric_definition_gate_pass_count"] == 1
    assert (
        audit["metric_definition_ranking_relevant_assessment_count"]
        == 0
    )
    assert (
        audit["metric_definition_ranking_relevant_gate_pass_count"]
        == 0
    )
    assert (
        audit["metric_definition_ranking_relevant_gate_blocked_count"]
        == 0
    )
    assert audit["passes_structural_gate"] is True


def test_alpha4b3b4c1_ranking_relevance_counts_enabled_policy():
    left = _context("c1", "P1", "m1")
    right = _context("c2", "P2", "m2")
    enabled = _assessment(left, right)

    metric_rows = build_metric_definition_assessments(
        comparison_assessments=[enabled],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            _metric(
                "d1",
                "P1",
                "m1",
                family="molecule_normalized_intensity_ratio",
                normalization="molecule_count",
                aggregation="single_particle",
            ),
            _metric(
                "d2",
                "P2",
                "m2",
                family="molecule_normalized_intensity_ratio",
                normalization="molecule_count",
                aggregation="single_particle",
            ),
        ],
        adapter=_metric_adapter(),
    )
    final = apply_metric_definition_numeric_gate(
        [enabled],
        metric_rows,
    )

    audit = audit_comparison_outputs(
        contexts=[left, right],
        assessments=final,
        source_graphs=_graphs(),
        adapter=_comparison_adapter(),
        metric_definition_assessments=metric_rows,
    )

    assert (
        audit["metric_definition_ranking_relevant_assessment_count"]
        == 1
    )
    assert (
        audit["metric_definition_ranking_relevant_gate_pass_count"]
        == 1
    )
    assert (
        audit["metric_definition_ranking_relevant_gate_blocked_count"]
        == 0
    )
    assert audit["passes_structural_gate"] is True


def test_alpha4b3b4c1_quality_gate_semantics_version():
    assert (
        QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID
        == "quality_aware_numeric_gate_v2_alpha4b3b4c1"
    )
