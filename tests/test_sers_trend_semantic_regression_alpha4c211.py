from __future__ import annotations

import networkx as nx

from domains.sers.trend_alpha4c211 import (
    SERS_AU_AG_TREND_ADAPTER,
    SERS_AU_AG_TREND_SEMANTICS_ID,
)
from domains.sers.trend_precision_alpha4c211 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER,
    SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
)
from dac_her.trend_domain import TrendEvidenceSource


_DUMMY = ({"fixture": True},)


def _claim_source(
    text: str,
    *,
    claim_id: str = "claim",
    subject_id: str = "",
    subject_label: str = "",
) -> TrendEvidenceSource:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(claim_id, type="ObservationClaim", statement=text)
    if subject_id:
        graph.add_node(
            subject_id,
            type="PlasmonicSubstrate",
            label=subject_label,
        )
        graph.add_edge(claim_id, subject_id, relation="APPLIES_TO")
    return TrendEvidenceSource(
        graph=graph,
        paper_id="fixture_paper",
        measurement_result_rows=_DUMMY,
        method_context_rows=_DUMMY,
        comparison_context_rows=_DUMMY,
    )


def _only(text: str):
    rows = SERS_AU_AG_TREND_ADAPTER.extract_evidence(
        _claim_source(text)
    )
    assert len(rows) == 1
    return rows[0]


def test_actual_sers1_optimum_claim_is_single_optimum():
    # Kiwook_SERS_1 / claim_optimal_agno3
    row = _only(
        "SERS intensity increased with AgNO3 concentration from 50 to "
        "300 mM and was highest at 300 mM, whereas signals decreased "
        "at 400 and 500 mM."
    )
    assert row.independent_variable_key == "silver_precursor_concentration"
    assert row.dependent_observable_key == "raman_intensity"
    assert row.direction == "non_monotonic"
    assert row.shape == "single_optimum"


def test_actual_sers5_presence_claim_is_not_nanogap_size():
    # Kiwook_SERS_5 / claim_interior_nanogap_enhancement
    row = _only(
        "DIPs produced markedly stronger SERS enhancement than the "
        "compared nanogap-less or shell-less nanoparticles when the "
        "Raman reporter was positioned in the interior nanogap."
    )
    assert row.independent_variable_key == "nanogap_presence"
    assert row.independent_variable_key != "nanogap_size"
    assert row.direction == "positive"
    assert row.shape == "unspecified"


def test_actual_sers8_formal_ef_direction_wins_over_generic_signal_language():
    # Source wording represented by Kiwook_SERS_8 / claim_ef_increases_gold.
    row = _only(
        "As the concentration of HAuCl4 increased, the SERS EF "
        "coefficient increased correspondingly in all Raman bands."
    )
    assert row.independent_variable_key == "gold_precursor_amount"
    assert row.dependent_observable_key == "sers_enhancement_factor"
    assert row.direction == "positive"


def test_actual_sers6_plural_ratios_and_landmark_normalization():
    # Kiwook_SERS_6 / claim_ratio_10_7_highest_sers
    text = (
        "Among the tested Au-Ag ratios, the 10:7 bimetallic "
        "nanoparticle substrate produced the strongest SERRS signal "
        "for methylene blue at 1622.1 cm^-1."
    )
    source = _claim_source(text)
    rows = SERS_AU_AG_TREND_ADAPTER.extract_evidence(source)
    assert len(rows) == 1
    row = rows[0]
    assert row.independent_variable_key == "ag_to_au_ratio"
    assert row.dependent_observable_key == "raman_intensity"
    assert row.direction == "non_monotonic"
    assert row.shape == "single_optimum"

    annotation = SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(
        row.to_row(),
        source.graph,
    )
    assert annotation.source_control_value_text == "10:7"
    assert annotation.canonical_control_value_numeric == 0.7
    assert annotation.normalization_transform == "au_ag_to_ag_over_au"


def test_actual_sers1_raman_peak_intensity_syntax_is_supported():
    # Kiwook_SERS_1 / claim_atp_detection
    row = _only(
        "SiO2@Au@Ag nanoparticles enabled label-free ATP detection "
        "over the reported concentration range, with Raman peak "
        "intensity increasing with ATP concentration."
    )
    assert row.independent_variable_key == "analyte_concentration"
    assert row.dependent_observable_key == "raman_intensity"
    assert row.direction == "positive"
    assert row.shape == "monotonic"


def test_actual_sers10_shell_claims_collapse_by_structural_family():
    # Kiwook_SERS_10 / claim_shell_thickness_trend
    claim_a = (
        "SERS intensity increased with Ag shell thickness from 3.6 to "
        "8.4 nm and reached an approximately optimal value at 8.4 nm; "
        "increasing the shell to 10.0 nm produced essentially the same "
        "enhancement factor of 5.8 relative to Au nanocubes."
    )
    # Kiwook_SERS_10 / claim_sers_shell_thickness
    claim_b = (
        "The SERS performance of Au@Ag nanocubes is significantly "
        "enhanced by controlling the Ag shell thickness; the Raman "
        "intensity increases with Ag shell thickness and approaches "
        "its maximum once the shell exceeds the critical thickness "
        "of approximately 8.4 nm."
    )
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node("claim_a", type="ObservationClaim", statement=claim_a)
    graph.add_node("claim_b", type="ObservationClaim", statement=claim_b)
    graph.add_node(
        "substrate_au_ag_10_0",
        type="PlasmonicSubstrate",
        label="Au@Ag10.0",
    )
    graph.add_node(
        "substrate_au55_ag8_4",
        type="PlasmonicSubstrate",
        label="Au55@Ag8.4",
    )
    graph.add_edge(
        "claim_a",
        "substrate_au_ag_10_0",
        relation="APPLIES_TO",
    )
    graph.add_edge(
        "claim_b",
        "substrate_au55_ag8_4",
        relation="APPLIES_TO",
    )
    source = TrendEvidenceSource(
        graph=graph,
        paper_id="fixture_paper",
        measurement_result_rows=_DUMMY,
        method_context_rows=_DUMMY,
        comparison_context_rows=_DUMMY,
    )
    evidence = SERS_AU_AG_TREND_ADAPTER.extract_evidence(source)
    assert len(evidence) == 2
    assert {
        (row.dependent_observable_key, row.direction, row.shape)
        for row in evidence
    } == {("raman_intensity", "positive", "saturating")}

    evidence_rows = [row.to_row() for row in evidence]
    annotations = [
        SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(row, graph)
        for row in evidence_rows
    ]
    results = SERS_AU_AG_TREND_PRECISION_ADAPTER.consolidate(
        evidence_rows,
        annotations,
        {"fixture_paper": graph},
    )
    assert len(results) == 1
    assert results[0].support_mention_count == 2


def test_calculated_numeric_ef_is_model_derived_not_formal_empirical_ef():
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "calc_dda_nanobox_gap_comparison",
        type="Calculation",
        label="DDA gap calculation",
    )
    row = {
        "trend_id": "trend:dda",
        "paper_id": "fixture_paper",
        "evidence_basis": "controlled_numeric_pair",
        "independent_variable_key": "nanogap_size",
        "dependent_observable_key": "sers_enhancement_factor",
        "source_expression": "",
        "source_expressions": [],
        "source_calculation_ids": ["calc_dda_nanobox_gap_comparison"],
        "source_measurement_ids": [],
        "source_node_ids": ["calc_dda_nanobox_gap_comparison"],
        "subject_ids": [],
    }
    annotation = SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(
        row,
        graph,
    )
    assert annotation.evidence_kind == "calculated_numeric"
    assert (
        annotation.observable_semantics
        == "model_derived_sers_enhancement_factor"
    )
    assert annotation.observable_semantics != "formal_sers_enhancement_factor"


def test_shell_relative_5_8_regression_still_maps_to_intensity():
    row = _only(
        "SERS intensity increased with Ag shell thickness from 3.6 to "
        "8.4 nm and reached an approximately optimal value at 8.4 nm; "
        "increasing the shell to 10.0 nm produced essentially the same "
        "enhancement factor of 5.8 relative to Au nanocubes."
    )
    assert row.dependent_observable_key == "raman_intensity"
    assert row.direction == "positive"
    assert row.shape == "saturating"


def test_alpha4c211_semantics_are_versioned():
    assert SERS_AU_AG_TREND_SEMANTICS_ID == "sers_au_ag_trend_v3_alpha4c211"
    assert (
        SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID
        == "sers_au_ag_trend_precision_v2_alpha4c211"
    )
