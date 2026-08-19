from __future__ import annotations

import networkx as nx

from domains.sers.trend_alpha4c2121 import (
    SERS_AU_AG_TREND_ADAPTER,
    SERS_AU_AG_TREND_SEMANTICS_ID,
    _canonical_claim_text_bundle,
    _is_spectral_alignment_claim,
    _reground_item,
)
from domains.sers.trend_precision_alpha4c2121 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER,
    SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
)
from dac_her.trend_domain import TrendEvidence


CLAIM_ID = "mech_lspr_laser_matching_increases_ef"


def _canonical_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        CLAIM_ID,
        type="MechanismClaim",
        claim_type="resonant_plasmonic_enhancement",
        basis="experimental",
        label=(
            "Ag–Au SERS enhancement increases when the "
            "surface-plasmon-resonance wavelength is closer to the "
            "excitation laser wavelength."
        ),
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


def _historical_bad_item() -> TrendEvidence:
    return TrendEvidence(
        trend_id="trend:historical-bad-axis",
        domain_profile_id="sers_au_ag",
        trend_semantics_id="sers_au_ag_trend_v4_alpha4c212",
        paper_id="Kiwook_SERS_8",
        independent_variable_key="excitation_wavelength",
        independent_variable_label="excitation wavelength",
        dependent_observable_key="raman_intensity",
        dependent_observable_label="Raman intensity",
        direction="positive",
        shape="monotonic",
        evidence_basis="reported_directional_claim",
        causal_status="not_asserted",
        varied_dimension="excitation_wavelength",
        subject_ids=("substrate_ag_au_alloy",),
        source_expression=(
            "Ag–Au SERS enhancement increases when the "
            "surface-plasmon-resonance wavelength is closer to the "
            "excitation laser wavelength."
        ),
        source_expressions=(
            "Ag–Au SERS enhancement increases when the "
            "surface-plasmon-resonance wavelength is closer to the "
            "excitation laser wavelength.",
        ),
        source_claim_ids=(CLAIM_ID,),
        source_node_ids=(CLAIM_ID,),
    )


def test_hyphenated_surface_plasmon_resonance_is_detected():
    graph = _canonical_graph()
    item = _historical_bad_item()
    bundle = _canonical_claim_text_bundle(item, graph)
    assert "surface-plasmon-resonance" in bundle
    assert _is_spectral_alignment_claim(bundle)


def test_canonical_claim_regrounds_bad_axis_and_formal_ef():
    graph = _canonical_graph()
    item = _historical_bad_item()

    refined = _reground_item(item, graph)

    assert refined.trend_semantics_id == (
        "sers_au_ag_trend_v5_alpha4c2121"
    )
    assert refined.independent_variable_key == (
        "spr_excitation_detuning"
    )
    assert refined.dependent_observable_key == (
        "sers_enhancement_factor"
    )
    assert refined.direction == "negative"
    assert refined.shape == "monotonic"
    assert refined.causal_status == "not_asserted"
    assert refined.varied_dimension == "spr_excitation_detuning"
    assert refined.source_claim_ids == (CLAIM_ID,)

    # Provenance wording is not silently rewritten.
    assert refined.source_expression == item.source_expression
    assert refined.source_expressions == item.source_expressions


def test_non_spectral_claim_is_semantics_only_upgrade():
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "claim_other",
        type="ObservationClaim",
        statement="Gold precursor amount increases SERS enhancement.",
    )
    item = TrendEvidence(
        trend_id="trend:other",
        domain_profile_id="sers_au_ag",
        trend_semantics_id="sers_au_ag_trend_v4_alpha4c212",
        paper_id="P",
        independent_variable_key="gold_precursor_amount",
        independent_variable_label="gold precursor amount",
        dependent_observable_key="sers_enhancement_factor",
        dependent_observable_label="SERS enhancement factor",
        direction="positive",
        shape="monotonic",
        evidence_basis="reported_directional_claim",
        source_expression="Gold precursor amount increases SERS enhancement.",
        source_expressions=(
            "Gold precursor amount increases SERS enhancement.",
        ),
        source_claim_ids=("claim_other",),
        source_node_ids=("claim_other",),
    )

    refined = _reground_item(item, graph)

    assert refined.trend_id == item.trend_id
    assert refined.independent_variable_key == (
        item.independent_variable_key
    )
    assert refined.dependent_observable_key == (
        item.dependent_observable_key
    )
    assert refined.direction == item.direction
    assert refined.trend_semantics_id == (
        "sers_au_ag_trend_v5_alpha4c2121"
    )


def test_semantics_ids():
    assert SERS_AU_AG_TREND_SEMANTICS_ID == (
        "sers_au_ag_trend_v5_alpha4c2121"
    )
    assert SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID == (
        "sers_au_ag_trend_precision_v4_alpha4c2121"
    )
    assert SERS_AU_AG_TREND_ADAPTER.semantics_id == (
        "sers_au_ag_trend_v5_alpha4c2121"
    )
    assert SERS_AU_AG_TREND_PRECISION_ADAPTER.trend_semantics_id == (
        "sers_au_ag_trend_v5_alpha4c2121"
    )
