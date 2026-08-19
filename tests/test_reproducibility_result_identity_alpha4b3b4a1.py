from __future__ import annotations

import networkx as nx

from domains.sers.reproducibility import (
    SERS_AU_AG_REPRODUCIBILITY_ADAPTER,
)
from dac_her.reproducibility_evidence import (
    audit_reproducibility_evidence,
)


def _base_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "sub",
        type="PlasmonicSubstrate",
        label="SERS substrate",
    )
    graph.add_node(
        "analyte",
        type="Analyte",
        label="ATP",
    )
    return graph


def _add_sers_experiment(
    graph: nx.MultiDiGraph,
    experiment_id: str,
) -> None:
    graph.add_node(
        experiment_id,
        type="Experiment",
        label="ATP SERS reproducibility",
        experiment_type="sers_spectroscopy",
        method_label="Surface-enhanced Raman spectroscopy",
        conditions_json="[]",
    )
    graph.add_edge("sub", experiment_id, relation="TESTED_IN")
    graph.add_edge(experiment_id, "analyte", relation="USES_ANALYTE")


def _add_rsd(
    graph: nx.MultiDiGraph,
    *,
    measurement_id: str,
    experiment_id: str,
    value_numeric: str,
    value_text: str,
    source_expression: str,
) -> None:
    graph.add_node(
        measurement_id,
        type="Measurement",
        metric_id="relative_standard_deviation",
        label="Relative standard deviation",
        subject_id="sub",
        value_numeric=value_numeric,
        value_text=value_text,
        unit="%" if value_numeric else "",
        source_expression=source_expression,
    )
    graph.add_edge(
        experiment_id,
        measurement_id,
        relation="HAS_MEASUREMENT",
    )
    graph.add_edge(
        measurement_id,
        "sub",
        relation="MEASURED_FOR",
    )


def test_alpha4b3b4a1_mislabeled_rsd_without_dispersion_is_repeatability():
    graph = _base_graph()
    _add_sers_experiment(graph, "exp")
    _add_rsd(
        graph,
        measurement_id="m",
        experiment_id="exp",
        value_numeric="",
        value_text="Reproducible Raman signals were obtained.",
        source_expression=(
            "Reproducible Raman signals were obtained in "
            "batch-to-batch experiments."
        ),
    )

    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(
        graph,
        "P1",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.evidence_kind == "repeatability_statement"
    assert row.value_numeric is None
    assert row.value_text == ""
    assert row.reproducibility_scope == "batch_to_batch"


def test_alpha4b3b4a1_explicit_textual_deviation_keeps_rsd_kind():
    graph = _base_graph()
    _add_sers_experiment(graph, "exp")
    _add_rsd(
        graph,
        measurement_id="m",
        experiment_id="exp",
        value_numeric="",
        value_text="14.2% deviation",
        source_expression="A 14.2% deviation was reported.",
    )

    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(
        graph,
        "P1",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.evidence_kind == "relative_standard_deviation"
    assert row.value_numeric is None
    assert row.value_text == "14.2% deviation"


def test_alpha4b3b4a1_shared_producer_exact_mentions_are_consolidated():
    graph = _base_graph()
    _add_sers_experiment(graph, "exp")
    for measurement_id, expression in (
        ("m1", "Batch-to-batch deviation was 14.2%."),
        ("m2", "The reported batch-to-batch RSD was 14.2%."),
    ):
        _add_rsd(
            graph,
            measurement_id=measurement_id,
            experiment_id="exp",
            value_numeric="14.2",
            value_text="",
            source_expression=expression,
        )

    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(
        graph,
        "P1",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.evidence_kind == "relative_standard_deviation"
    assert row.result_identity_status == "consolidated_exact"
    assert set(row.source_mention_node_ids) == {"m1", "m2"}
    assert len(row.source_expressions) == 2
    assert {"m1", "m2", "exp"} <= set(row.source_node_ids)

    audit = audit_reproducibility_evidence(
        evidence=rows,
        source_graphs={"P1": graph},
        adapter=SERS_AU_AG_REPRODUCIBILITY_ADAPTER,
    )
    assert audit.structural_gate is True
    assert audit.source_mention_count == 2
    assert audit.consolidated_result_count == 1
    assert audit.possible_duplicate_result_pair_count == 0


def test_alpha4b3b4a1_same_lineage_qualitative_and_rsd_are_one_result():
    graph = _base_graph()
    _add_sers_experiment(graph, "exp")
    _add_rsd(
        graph,
        measurement_id="m_quality",
        experiment_id="exp",
        value_numeric="",
        value_text="Reproducible Raman signals were obtained.",
        source_expression=(
            "Reproducible Raman signals were obtained in "
            "batch-to-batch experiments."
        ),
    )
    _add_rsd(
        graph,
        measurement_id="m_rsd",
        experiment_id="exp",
        value_numeric="14.2",
        value_text="",
        source_expression="Batch-to-batch deviation was 14.2%.",
    )

    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(
        graph,
        "P1",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.evidence_kind == "relative_standard_deviation"
    assert row.value_numeric == 14.2
    assert row.result_identity_status == "consolidated_exact"
    assert set(row.source_mention_node_ids) == {"m_quality", "m_rsd"}
    assert len(row.source_expressions) == 2


def test_alpha4b3b4a1_same_value_without_shared_lineage_is_candidate_not_merge():
    graph = _base_graph()
    for experiment_id, measurement_id in (
        ("exp1", "m1"),
        ("exp2", "m2"),
    ):
        _add_sers_experiment(graph, experiment_id)
        _add_rsd(
            graph,
            measurement_id=measurement_id,
            experiment_id=experiment_id,
            value_numeric="14.2",
            value_text="",
            source_expression=(
                "Batch-to-batch Raman-signal deviation was 14.2%."
            ),
        )

    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(
        graph,
        "P1",
    )
    assert len(rows) == 2
    assert all(
        row.result_identity_status == "single_mention"
        for row in rows
    )

    audit = audit_reproducibility_evidence(
        evidence=rows,
        source_graphs={"P1": graph},
        adapter=SERS_AU_AG_REPRODUCIBILITY_ADAPTER,
    )
    assert audit.structural_gate is True
    assert audit.possible_duplicate_result_pair_count == 1
    assert audit.possible_duplicate_result_cluster_count == 1
    assert (
        audit.possible_duplicate_results[0].reason
        == "same_quantitative_result_without_shared_lineage"
    )


def test_alpha4b3b4a1_different_subjects_are_not_duplicate_candidates():
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node("sub1", type="PlasmonicSubstrate")
    graph.add_node("sub2", type="PlasmonicSubstrate")
    graph.add_node("analyte", type="Analyte", label="ATP")
    for index in (1, 2):
        exp = f"exp{index}"
        meas = f"m{index}"
        sub = f"sub{index}"
        graph.add_node(
            exp,
            type="Experiment",
            label="SERS reproducibility",
            experiment_type="sers_spectroscopy",
            method_label="SERS",
            conditions_json="[]",
        )
        graph.add_edge(sub, exp, relation="TESTED_IN")
        graph.add_edge(exp, "analyte", relation="USES_ANALYTE")
        graph.add_node(
            meas,
            type="Measurement",
            metric_id="relative_standard_deviation",
            subject_id=sub,
            value_numeric="14.2",
            value_text="",
            unit="%",
            source_expression="Batch-to-batch deviation was 14.2%.",
        )
        graph.add_edge(exp, meas, relation="HAS_MEASUREMENT")
        graph.add_edge(meas, sub, relation="MEASURED_FOR")

    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(
        graph,
        "P1",
    )
    audit = audit_reproducibility_evidence(
        evidence=rows,
        source_graphs={"P1": graph},
        adapter=SERS_AU_AG_REPRODUCIBILITY_ADAPTER,
    )
    assert len(rows) == 2
    assert audit.possible_duplicate_result_pair_count == 0


def test_alpha4b3b4a1_semantics_version():
    assert (
        SERS_AU_AG_REPRODUCIBILITY_ADAPTER.semantics_id
        == "sers_au_ag_reproducibility_v2_alpha4b3b4a1"
    )
