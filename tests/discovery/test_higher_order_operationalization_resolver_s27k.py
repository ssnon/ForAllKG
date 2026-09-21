from __future__ import annotations

import networkx as nx

from pipeline_core.discovery.higher_order_operationalization_resolver import (
    resolve_operationalization_witnesses,
)
from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirement,
    OperationalizationWitnessRequirementSet,
)


def _requirement(
    *,
    rid: str = "req:1",
    target: str = "electromagnetic hotspot location and intensity",
    anchor: str = "SERS enhancement factor",
):
    row = OperationalizationWitnessRequirement(
        requirement_id=rid,
        source_experiment_id="exp:source",
        output_experiment_id="exp:out",
        issue_code="MEASUREMENT_INDEPENDENCE_UNRESOLVED",
        status="unresolved_requires_measurement_witness",
        requested_target_observable=target,
        anchor_observable=anchor,
        observable_pair=[target, anchor],
        required_graph_topology=["test"],
        minimum_candidate_evidence=["test"],
        independence_acceptance_rule="test",
        independence_rejection_rule="test",
    )
    return OperationalizationWitnessRequirementSet(
        repaired_experiment_count=1,
        requirement_count=1,
        candidate_inspiration_requirement_count=0,
        requirements=[row],
    )


def _measurement(
    graph,
    mid,
    metric,
    provider,
    *,
    paper="paper:1",
):
    graph.add_node(
        mid,
        type="Measurement",
        metric=metric,
        metric_id=metric.lower().replace(" ", "_"),
        label=metric,
        source_expression=f"reported {metric}",
        description="",
        subject_id="sample:1",
        source_paper_id=paper,
    )
    graph.add_edge(
        provider,
        mid,
        relation="HAS_MEASUREMENT",
        paper_id=paper,
        evidence_text=f"measured {metric}",
    )


def test_distinct_providers_and_methods_create_candidate_not_verification():
    g = nx.MultiDiGraph()
    g.add_node(
        "calc:field",
        type="Calculation",
        label="FDTD field calculation",
        calculation_type="FDTD",
        method_details="finite-difference time-domain",
        source_paper_id="paper:t",
    )
    g.add_node(
        "exp:sers",
        type="Experiment",
        label="SERS spectroscopy",
        method_label="surface-enhanced Raman spectroscopy",
        raw_method_name="SERS",
        experiment_family="spectroscopy",
        source_paper_id="paper:a",
    )
    _measurement(
        g,
        "m:target",
        "hotspot intensity",
        "calc:field",
        paper="paper:t",
    )
    _measurement(
        g,
        "m:anchor",
        "SERS enhancement factor",
        "exp:sers",
        paper="paper:a",
    )

    result = resolve_operationalization_witnesses(
        graph=g,
        requirements=_requirement(),
    )

    row = result.resolutions[0]
    assert row.status == "SUPPORTED_DISTINCT_OPERATIONALIZATION"
    assert row.measurement_independence_verified is False
    assert result.measurement_independence_verified_count == 0
    assert row.pair_candidates[0].distinct_provider_ids_present is True
    assert row.pair_candidates[0].distinct_method_identity_present is True


def test_same_provider_remains_unresolved():
    g = nx.MultiDiGraph()
    g.add_node(
        "exp:one",
        type="Experiment",
        label="combined optical experiment",
        method_label="optical spectroscopy",
        raw_method_name="combined measurement",
        experiment_family="spectroscopy",
        source_paper_id="paper:1",
    )
    _measurement(
        g,
        "m:target",
        "hotspot intensity",
        "exp:one",
    )
    _measurement(
        g,
        "m:anchor",
        "SERS enhancement factor",
        "exp:one",
    )

    result = resolve_operationalization_witnesses(
        graph=g,
        requirements=_requirement(),
    )

    assert result.resolutions[0].status == (
        "SAME_PROVIDER_OR_METHOD_UNRESOLVED"
    )


def test_only_anchor_found_is_classified_fail_closed():
    g = nx.MultiDiGraph()
    g.add_node(
        "exp:sers",
        type="Experiment",
        label="SERS spectroscopy",
        method_label="surface-enhanced Raman spectroscopy",
        raw_method_name="SERS",
        experiment_family="spectroscopy",
        source_paper_id="paper:1",
    )
    _measurement(
        g,
        "m:anchor",
        "SERS enhancement factor",
        "exp:sers",
    )

    result = resolve_operationalization_witnesses(
        graph=g,
        requirements=_requirement(),
    )

    assert result.resolutions[0].status == "ONLY_ANCHOR_FOUND"
    assert result.resolutions[0].target_candidate_count == 0


def test_missing_provider_identity_does_not_become_supported():
    g = nx.MultiDiGraph()
    g.add_node(
        "m:target",
        type="Measurement",
        metric="hotspot intensity",
        metric_id="hotspot_intensity",
        label="hotspot intensity",
        source_expression="hotspot intensity distribution",
        description="",
        subject_id="sample:1",
        source_paper_id="paper:1",
    )
    g.add_node(
        "m:anchor",
        type="Measurement",
        metric="SERS enhancement factor",
        metric_id="sers_enhancement_factor",
        label="SERS enhancement factor",
        source_expression="SERS enhancement factor",
        description="",
        subject_id="sample:1",
        source_paper_id="paper:1",
    )

    result = resolve_operationalization_witnesses(
        graph=g,
        requirements=_requirement(),
    )

    row = result.resolutions[0]
    assert row.status == "PROVIDER_IDENTITY_UNRESOLVED"
    assert row.measurement_independence_verified is False
    assert result.independence_certification_authority is False
