from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from dac_her.measurement_result_identity import (
    MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
    audit_measurement_result_identities,
    build_measurement_result_identities,
)


def _subject(
    graph: nx.MultiDiGraph,
    node_id: str,
    label: str,
):
    graph.add_node(
        node_id,
        type="PlasmonicSubstrate",
        label=label,
        description="",
    )


def _measurement(
    graph: nx.MultiDiGraph,
    node_id: str,
    *,
    source_local_id: str = "",
    subject_id: str,
    value: float = 2.4,
    unit: str = "nM",
    conditions=None,
    qualifier: str = "",
    collision: bool = False,
):
    graph.add_node(
        node_id,
        type="Measurement",
        label="Detection limit",
        metric_id="detection_limit",
        metric="Detection limit",
        subject_id=subject_id,
        source_expression="LOD of 2.4 nM for ATP",
        source_local_id=source_local_id,
        id_collision_reason=(
            "measurement_payload_conflict" if collision else ""
        ),
        value_numeric=value,
        value_text="",
        unit=unit,
        qualifier=qualifier,
        basis="",
        conditions_json=__import__("json").dumps(
            conditions
            or [
                {
                    "name": "analyte",
                    "value_numeric": None,
                    "value_text": "ATP",
                    "unit": None,
                    "reference": None,
                }
            ]
        ),
    )
    graph.add_edge(
        node_id,
        subject_id,
        relation="MEASURED_FOR",
    )


def test_exact_duplicate_source_mentions_consolidate():
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    _subject(
        graph,
        "substrate_a",
        "SiO2@Au@Ag nanoparticles",
    )
    _subject(
        graph,
        "substrate_b",
        "SiO2@Au@Ag NPs for ATP detection",
    )
    _measurement(
        graph,
        "meas_lod_atp",
        subject_id="substrate_a",
        qualifier="theoretical",
        conditions=[
            {
                "name": "analyte",
                "value_numeric": None,
                "value_text": "ATP",
                "unit": None,
                "reference": None,
            },
            {
                "name": "Raman shift",
                "value_numeric": 1575,
                "value_text": None,
                "unit": "cm^-1",
                "reference": None,
            },
        ],
    )
    _measurement(
        graph,
        "meas_lod_atp__mention_measurement_deadbeef",
        source_local_id="meas_lod_atp",
        subject_id="substrate_b",
        collision=True,
    )

    identities, candidates = build_measurement_result_identities(
        graph,
        "P1",
    )
    assert len(identities) == 1
    identity = identities[0]
    assert identity.status == "consolidated_exact"
    assert identity.representative_measurement_id == "meas_lod_atp"
    assert set(identity.source_mention_ids) == {
        "meas_lod_atp",
        "meas_lod_atp__mention_measurement_deadbeef",
    }
    assert candidates[0].exact_consolidation_allowed is True

    audit = audit_measurement_result_identities(
        identities=identities,
        candidates=candidates,
        source_graphs={"P1": graph},
    )
    assert audit.source_mention_count == 2
    assert audit.scientific_result_count == 1
    assert audit.consolidated_exact_result_count == 1
    assert audit.structural_gate is True


def test_same_value_alone_never_merges_without_same_origin_lineage():
    graph = nx.MultiDiGraph()
    _subject(graph, "s", "SiO2@Au@Ag nanoparticles")
    _measurement(graph, "m1", subject_id="s")
    _measurement(graph, "m2", subject_id="s")
    identities, _ = build_measurement_result_identities(
        graph,
        "P1",
    )
    assert len(identities) == 2
    assert all(row.status == "single_mention" for row in identities)


def test_explicit_condition_conflict_blocks_consolidation():
    graph = nx.MultiDiGraph()
    _subject(graph, "s", "SiO2@Au@Ag nanoparticles")
    _measurement(
        graph,
        "m",
        subject_id="s",
        conditions=[
            {
                "name": "Raman shift",
                "value_numeric": 1575,
                "value_text": None,
                "unit": "cm^-1",
                "reference": None,
            }
        ],
    )
    _measurement(
        graph,
        "m__mention_measurement_x",
        source_local_id="m",
        subject_id="s",
        collision=True,
        conditions=[
            {
                "name": "Raman shift",
                "value_numeric": 1078,
                "value_text": None,
                "unit": "cm^-1",
                "reference": None,
            }
        ],
    )
    identities, candidates = build_measurement_result_identities(
        graph,
        "P1",
    )
    assert len(identities) == 2
    assert candidates[0].exact_consolidation_allowed is False
    assert any(
        blocker.startswith("condition_mismatch:")
        for blocker in candidates[0].blockers
    )


def test_explicit_subject_structural_conflict_blocks_consolidation():
    graph = nx.MultiDiGraph()
    _subject(graph, "alloy", "Ag-Au alloy nanoplate")
    _subject(graph, "core_shell", "Ag-Au core shell nanoplate")
    _measurement(graph, "m", subject_id="alloy")
    _measurement(
        graph,
        "m__mention_measurement_x",
        source_local_id="m",
        subject_id="core_shell",
        collision=True,
    )
    identities, candidates = build_measurement_result_identities(
        graph,
        "P1",
    )
    assert len(identities) == 2
    assert candidates[0].exact_consolidation_allowed is False
    assert "subject_structural_role_mismatch" in candidates[0].blockers


def test_identity_semantics_version():
    assert (
        MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID
        == "measurement_result_identity_v1_alpha4b4a1"
    )
