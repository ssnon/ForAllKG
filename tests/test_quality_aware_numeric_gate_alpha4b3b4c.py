from __future__ import annotations

import networkx as nx
import pytest

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


def _comparison_context(
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
) -> ComparisonAssessment:
    return ComparisonAssessment(
        assessment_id="comparison:test",
        comparison_semantics_id="cmp-v1",
        observable_key=left.observable_key,
        observable_policy_id="policy",
        observable_family="sers_performance",
        applicable_dimensions=("analyte",),
        ranking_required_dimensions=("analyte",),
        numeric_ranking_mode="allowed_if_complete",
        ranking_direction="higher_better",
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
        numeric_ranking_allowed=True,
    )


def _metric(
    context_id: str,
    paper_id: str,
    measurement_id: str,
    *,
    status: str = "known",
    family: str = "molecule_normalized_intensity_ratio",
    aggregation: str = "single_particle",
    normalization: str = "molecule_count",
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
        definition_status=status,
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
            "reported_ef_unspecified",
        }),
        aggregation_scopes=frozenset({
            "single_particle",
            "population_mean",
            "unspecified",
        }),
        normalization_bases=frozenset({
            "molecule_count",
            "concentration",
            "unspecified",
        }),
        reference_bases=frozenset({
            "normal_raman",
            "unspecified",
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


def _graph(measurement_id: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        measurement_id,
        type="Measurement",
    )
    return graph


def test_alpha4b3b4c_same_known_definition_keeps_numeric_gate_open():
    left = _comparison_context("c1", "P1", "m1")
    right = _comparison_context("c2", "P2", "m2")
    base = _assessment(left, right)

    metric_rows = build_metric_definition_assessments(
        comparison_assessments=[base],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            _metric("d1", "P1", "m1"),
            _metric("d2", "P2", "m2"),
        ],
        adapter=_metric_adapter(),
    )
    assert len(metric_rows) == 1
    metric = metric_rows[0]
    assert metric.compatibility == "same_definition"
    assert metric.numeric_metric_definition_gate is True

    final = apply_metric_definition_numeric_gate(
        [base],
        metric_rows,
    )[0]
    assert final.numeric_ranking_allowed is True
    assert final.metric_definition_gate is True
    assert (
        final.metric_definition_compatibility
        == "same_definition"
    )


def test_alpha4b3b4c_unknown_definition_blocks_numeric_ranking():
    left = _comparison_context("c1", "P1", "m1")
    right = _comparison_context("c2", "P2", "m2")
    base = _assessment(left, right)

    unknown = _metric(
        "d1",
        "P1",
        "m1",
        status="unknown",
        family="reported_ef_unspecified",
        aggregation="unspecified",
        normalization="unspecified",
        reference="unspecified",
    )
    rows = build_metric_definition_assessments(
        comparison_assessments=[base],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            unknown,
            _metric("d2", "P2", "m2"),
        ],
        adapter=_metric_adapter(),
    )
    assert rows[0].compatibility == "unknown"
    assert rows[0].numeric_metric_definition_gate is False

    final = apply_metric_definition_numeric_gate(
        [base],
        rows,
    )[0]
    assert final.numeric_ranking_allowed is False
    assert "numeric_ranking_blocked_by_metric_definition" in (
        final.reasons
    )


def test_alpha4b3b4c_different_definition_family_blocks():
    left = _comparison_context("c1", "P1", "m1")
    right = _comparison_context("c2", "P2", "m2")
    base = _assessment(left, right)

    rows = build_metric_definition_assessments(
        comparison_assessments=[base],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            _metric("d1", "P1", "m1"),
            _metric(
                "d2",
                "P2",
                "m2",
                family="concentration_normalized_intensity_ratio",
                normalization="concentration",
            ),
        ],
        adapter=_metric_adapter(),
    )
    assert rows[0].compatibility == "different_definition"
    assert rows[0].numeric_metric_definition_gate is False


def test_alpha4b3b4c_unspecified_aggregation_blocks_even_when_definition_known():
    left = _comparison_context("c1", "P1", "m1")
    right = _comparison_context("c2", "P2", "m2")
    base = _assessment(left, right)

    rows = build_metric_definition_assessments(
        comparison_assessments=[base],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            _metric(
                "d1",
                "P1",
                "m1",
                aggregation="unspecified",
            ),
            _metric(
                "d2",
                "P2",
                "m2",
                aggregation="unspecified",
            ),
        ],
        adapter=_metric_adapter(),
    )
    assert rows[0].compatibility == "unknown"
    assert rows[0].numeric_metric_definition_gate is False
    assert "metric_aggregation_scope_unknown" in rows[0].reasons


def test_alpha4b3b4c_unsupported_observable_is_not_applicable_not_blocked():
    left = _comparison_context(
        "c1",
        "P1",
        "m1",
        observable="raman_intensity",
    )
    right = _comparison_context(
        "c2",
        "P2",
        "m2",
        observable="raman_intensity",
    )
    base = _assessment(left, right)

    rows = build_metric_definition_assessments(
        comparison_assessments=[base],
        comparison_contexts=[left, right],
        metric_definition_contexts=[],
        adapter=_metric_adapter(),
    )
    assert rows[0].compatibility == "not_applicable"
    assert rows[0].numeric_metric_definition_gate is True

    final = apply_metric_definition_numeric_gate(
        [base],
        rows,
    )[0]
    assert final.numeric_ranking_allowed is True
    assert final.metric_definition_compatibility == "not_applicable"


def test_alpha4b3b4c_missing_registered_metric_context_fails_closed():
    left = _comparison_context("c1", "P1", "m1")
    right = _comparison_context("c2", "P2", "m2")
    base = _assessment(left, right)

    with pytest.raises(ValueError, match="Missing MetricDefinitionContext"):
        build_metric_definition_assessments(
            comparison_assessments=[base],
            comparison_contexts=[left, right],
            metric_definition_contexts=[
                _metric("d1", "P1", "m1"),
            ],
            adapter=_metric_adapter(),
        )


def test_alpha4b3b4c_combined_audit_requires_metric_gate_for_numeric_ranking():
    left = _comparison_context("c1", "P1", "m1")
    right = _comparison_context("c2", "P2", "m2")
    base = _assessment(left, right)

    rows = build_metric_definition_assessments(
        comparison_assessments=[base],
        comparison_contexts=[left, right],
        metric_definition_contexts=[
            _metric(
                "d1",
                "P1",
                "m1",
                status="unknown",
                family="reported_ef_unspecified",
                aggregation="unspecified",
                normalization="unspecified",
                reference="unspecified",
            ),
            _metric("d2", "P2", "m2"),
        ],
        adapter=_metric_adapter(),
    )

    # Deliberately unsafe: link the gate metadata but keep numeric=True.
    unsafe = base.__class__(
        **{
            **base.__dict__,
            "metric_definition_assessment_id": rows[0].assessment_id,
            "metric_definition_compatibility": rows[0].compatibility,
            "metric_definition_gate": False,
        }
    )

    audit = audit_comparison_outputs(
        contexts=[left, right],
        assessments=[unsafe],
        source_graphs={
            "P1": _graph("m1"),
            "P2": _graph("m2"),
        },
        adapter=_comparison_adapter(),
        metric_definition_assessments=rows,
    )
    codes = {item["code"] for item in audit["issues"]}
    assert "UNSAFE_NUMERIC_RANKING_METRIC_DEFINITION" in codes
    assert "UNSAFE_NUMERIC_RANKING_METRIC_COMPATIBILITY" in codes
    assert audit["passes_structural_gate"] is False


def test_alpha4b3b4c_quality_gate_semantics_version():
    assert (
        QUALITY_AWARE_NUMERIC_GATE_SEMANTICS_ID
        == "quality_aware_numeric_gate_v2_alpha4b3b4c1"
    )
