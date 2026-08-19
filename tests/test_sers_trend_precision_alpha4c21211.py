from __future__ import annotations

import networkx as nx

from domains.sers.trend_precision_alpha4c21211 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER,
    SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
)


CLAIM_ID = "mech_lspr_laser_matching_increases_ef"


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        CLAIM_ID,
        type="MechanismClaim",
        statement=(
            "Ag–Au SERS enhancement increases when the "
            "surface-plasmon-resonance wavelength is closer to the "
            "excitation laser wavelength."
        ),
        description=(
            "The text presents proximity between the excitation laser "
            "wavelength and the Ag–Au nanoparticle SPR as associated "
            "with higher enhancement factor."
        ),
    )
    graph.add_node(
        "substrate_ag_au_alloy",
        type="PlasmonicSubstrate",
        label="Ag–Au alloy substrate",
    )
    return graph


def _regrounded_row():
    return {
        "trend_id": "trend:synthetic-spectral",
        "domain_profile_id": "sers_au_ag",
        "trend_semantics_id":
            "sers_au_ag_trend_v5_alpha4c2121",
        "paper_id": "P",
        "independent_variable_key": "spr_excitation_detuning",
        "independent_variable_label":
            "SPR–excitation spectral detuning",
        "dependent_observable_key": "sers_enhancement_factor",
        "dependent_observable_label": "SERS enhancement factor",
        "direction": "negative",
        "shape": "monotonic",
        "evidence_basis": "reported_directional_claim",
        "causal_status": "not_asserted",
        "varied_dimension": "spr_excitation_detuning",
        "subject_ids": ["substrate_ag_au_alloy"],
        "series_points": [],
        "source_expression": (
            "Ag–Au SERS enhancement increases when the "
            "surface-plasmon-resonance wavelength is closer to the "
            "excitation laser wavelength."
        ),
        "source_expressions": [
            "Ag–Au SERS enhancement increases when the "
            "surface-plasmon-resonance wavelength is closer to the "
            "excitation laser wavelength."
        ],
        "source_claim_ids": [CLAIM_ID],
        "source_measurement_ids": [],
        "source_measurement_group_ids": [],
        "source_experiment_ids": [],
        "source_calculation_ids": [],
        "source_measurement_result_ids": [],
        "source_method_context_ids": [],
        "source_comparison_context_ids": [],
        "source_node_ids": [CLAIM_ID],
        "requires_verification": False,
    }


def test_spectral_local_result_reconciles_to_active_annotation_semantics():
    graph = _graph()
    row = _regrounded_row()

    annotation = SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(
        row,
        graph,
    )
    assert annotation.control_family == "optical_alignment"
    assert annotation.observable_semantics == (
        "formal_sers_enhancement_factor"
    )

    results = SERS_AU_AG_TREND_PRECISION_ADAPTER.consolidate(
        [row],
        [annotation],
        {"P": graph},
    )
    assert len(results) == 1
    result = results[0]

    assert result.control_family == "optical_alignment"
    assert result.observable_semantics == (
        "formal_sers_enhancement_factor"
    )
    assert result.independent_variable_key == (
        "spr_excitation_detuning"
    )
    assert result.dependent_observable_key == (
        "sers_enhancement_factor"
    )


def test_precision_semantics_id():
    assert SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID == (
        "sers_au_ag_trend_precision_v5_alpha4c21211"
    )
    assert (
        SERS_AU_AG_TREND_PRECISION_ADAPTER.trend_semantics_id
        == "sers_au_ag_trend_v5_alpha4c2121"
    )
