from __future__ import annotations

import networkx as nx

from dac_her.domains.sers_au_ag_trend_alpha4c212 import (
    SERS_AU_AG_TREND_ADAPTER,
    SERS_AU_AG_TREND_SEMANTICS_ID,
)
from dac_her.domains.sers_au_ag_trend_precision_alpha4c212 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER,
    SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
)
from dac_her.trend_domain import TrendEvidenceSource


def _claim_only_source(graph: nx.Graph) -> TrendEvidenceSource:
    # SERS trend adapters intentionally require all four frozen inputs.
    # These harmless sentinel sidecars satisfy the generic source contract;
    # claim extraction itself does not consume them.
    return TrendEvidenceSource(
        graph=graph,
        paper_id="P",
        measurement_result_rows=({"sentinel": "identity"},),
        method_context_rows=({"sentinel": "method"},),
        comparison_context_rows=({"sentinel": "comparison"},),
    )


def test_spectral_matching_uses_detuning_axis():
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "mech_lspr_laser_matching_increases_ef",
        type="MechanismClaim",
        statement=(
            "The closer the SPR of the Ag-Au nanoparticles is to "
            "the exciting laser wavelength, the higher the "
            "enhancement factor will be."
        ),
    )
    evidence = SERS_AU_AG_TREND_ADAPTER.extract_evidence(
        _claim_only_source(graph)
    )
    matches = [
        item for item in evidence
        if item.independent_variable_key
            == "spr_excitation_detuning"
    ]
    assert len(matches) == 1
    assert matches[0].dependent_observable_key == (
        "sers_enhancement_factor"
    )
    assert matches[0].direction == "negative"
    assert matches[0].shape == "monotonic"
    assert matches[0].source_claim_ids == (
        "mech_lspr_laser_matching_increases_ef",
    )
    assert matches[0].causal_status == "not_asserted"
    assert not any(
        item.independent_variable_key == "excitation_wavelength"
        for item in evidence
    )


def test_spectral_matching_without_direction_remains_fail_closed():
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "claim_matching_matters",
        type="ObservationClaim",
        statement=(
            "SPR matching with the excitation laser is important "
            "for the SERS enhancement factor."
        ),
    )
    evidence = SERS_AU_AG_TREND_ADAPTER.extract_evidence(
        _claim_only_source(graph)
    )
    assert not any(
        item.source_claim_ids == ("claim_matching_matters",)
        and item.independent_variable_key
            == "spr_excitation_detuning"
        for item in evidence
    )


def _claim(
    trend_id: str,
    claim_id: str,
    subject_id: str,
    landmark: str = "8.4 nm",
):
    return {
        "trend_id": trend_id,
        "domain_profile_id": "sers_au_ag",
        "trend_semantics_id":
            "sers_au_ag_trend_v4_alpha4c212",
        "paper_id": "P",
        "independent_variable_key": "shell_thickness",
        "dependent_observable_key": "raman_intensity",
        "direction": "positive",
        "shape": "saturating",
        "evidence_basis": "reported_directional_claim",
        "source_expression": (
            "SERS intensity increases with Ag shell thickness and "
            f"approaches the maximum at {landmark}."
        ),
        "source_expressions": [],
        "source_claim_ids": [claim_id],
        "source_measurement_ids": [],
        "source_measurement_result_ids": [],
        "source_calculation_ids": [],
        "source_node_ids": [claim_id],
        "subject_ids": [subject_id],
    }


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "substrate_au_ag_10_0",
        type="PlasmonicSubstrate",
        label="Au@Ag core-shell nanocube",
    )
    graph.add_node(
        "substrate_au55_ag8_4",
        type="PlasmonicSubstrate",
        label="Au55@Ag8.4 core-shell nanocube",
    )
    graph.add_node(
        "substrate_au_ag_other",
        type="PlasmonicSubstrate",
        label="Au@Ag core-shell nanocube",
    )
    graph.add_node("c1", type="ObservationClaim")
    graph.add_node("c2", type="ObservationClaim")
    return graph


def test_structural_aliases_same_landmark_merge():
    graph = _graph()
    rows = [
        _claim("t1", "c1", "substrate_au_ag_10_0"),
        _claim("t2", "c2", "substrate_au55_ag8_4"),
    ]
    annotations = [
        SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(
            row,
            graph,
        )
        for row in rows
    ]
    results = (
        SERS_AU_AG_TREND_PRECISION_ADAPTER.consolidate(
            rows,
            annotations,
            {"P": graph},
        )
    )
    assert len(results) == 1
    assert results[0].support_mention_count == 2
    assert set(results[0].source_claim_ids) == {"c1", "c2"}


def test_different_landmarks_do_not_merge():
    graph = _graph()
    rows = [
        _claim(
            "t1",
            "c1",
            "substrate_au_ag_10_0",
            "8.4 nm",
        ),
        _claim(
            "t2",
            "c2",
            "substrate_au55_ag8_4",
            "10.0 nm",
        ),
    ]
    annotations = [
        SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(
            row,
            graph,
        )
        for row in rows
    ]
    assert {
        annotation.canonical_control_value_numeric
        for annotation in annotations
    } == {8.4, 10.0}

    results = (
        SERS_AU_AG_TREND_PRECISION_ADAPTER.consolidate(
            rows,
            annotations,
            {"P": graph},
        )
    )
    assert len(results) == 2
    assert all(
        result.support_mention_count == 1
        for result in results
    )


def test_same_landmark_different_structural_forms_do_not_merge():
    graph = _graph()
    graph.add_node(
        "substrate_au_ag_nanobox",
        type="PlasmonicSubstrate",
        label="Au/Ag nanobox",
    )
    rows = [
        _claim("t1", "c1", "substrate_au_ag_10_0"),
        _claim("t2", "c2", "substrate_au_ag_nanobox"),
    ]
    annotations = [
        SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(
            row,
            graph,
        )
        for row in rows
    ]
    results = (
        SERS_AU_AG_TREND_PRECISION_ADAPTER.consolidate(
            rows,
            annotations,
            {"P": graph},
        )
    )
    assert len(results) == 2


def test_semantics_ids():
    assert (
        SERS_AU_AG_TREND_SEMANTICS_ID
        == "sers_au_ag_trend_v4_alpha4c212"
    )
    assert (
        SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID
        == "sers_au_ag_trend_precision_v3_alpha4c212"
    )
