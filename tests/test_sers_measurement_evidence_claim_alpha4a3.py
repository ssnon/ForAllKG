from __future__ import annotations

import networkx as nx
from pathlib import Path

from pipeline_core.corpus.graph_semantics import evidence_topology_diagnostics
from domains.sers.prompts import (
    SERS_MICRO_REEXTRACT_SYSTEM_PROMPT,
    SERS_PATCH_SYSTEM_PROMPT,
    SERS_PROMPT_VERSION,
    SERS_SYSTEM_PROMPT,
)
from pipeline_core.corpus.vocab_registry import load_default_registries


def test_sers_prompt_version_alpha4a3():
    assert SERS_PROMPT_VERSION.startswith("sers-au-ag-extraction-v1-alpha4a")


def test_main_prompt_encodes_evidence_topology():
    prompt = SERS_SYSTEM_PROMPT
    normalized_prompt = " ".join(prompt.split())
    assert "EVIDENCE TOPOLOGY" in prompt
    assert "Scientific Entities are subjects, not evidence producers" in prompt
    assert "Experiment/Calculation --HAS_MEASUREMENT--> Measurement" in prompt
    assert "Measurement/Experiment/Calculation --SUPPORTS_CLAIM-->" in prompt
    assert "ObservationClaim --INTERPRETED_AS--> MechanismClaim" in prompt
    assert "Never use PlasmonicSubstrate" in normalized_prompt
    assert "SynthesisMethod" in normalized_prompt
    assert "as a SUPPORTS_CLAIM source merely because a claim is about that object" in normalized_prompt


def test_main_prompt_requires_complete_method_nodes():
    prompt = SERS_SYSTEM_PROMPT
    assert "METHOD NODE COMPLETENESS" in prompt
    assert "Every PREPARED_BY target" in prompt
    assert "must exist exactly once as a SynthesisMethod" in prompt
    assert "edge-only placeholder" in prompt


def test_patch_prompt_does_not_retype_subject_into_evidence():
    prompt = SERS_PATCH_SYSTEM_PROMPT
    assert "EVIDENCE-TOPOLOGY REPAIR" in prompt
    assert "Never retype a substrate, material, nanostructure, or SynthesisMethod" in prompt
    assert "HAS_MEASUREMENT" in prompt
    assert "unresolved_issue_ids" in prompt


def test_micro_prompt_reconstructs_measurement_and_claim_chain():
    prompt = SERS_MICRO_REEXTRACT_SYSTEM_PROMPT
    assert "EVIDENCE TOPOLOGY" in prompt
    assert "Measurement without an explicit source-grounded producer" in prompt
    assert "ObservationClaim requires" in prompt
    assert "MechanismClaim requires" in prompt
    assert "scientific subject is not itself evidence" in prompt


def test_pilot_vocabulary_calibration():
    experiments, metrics = load_default_registries(Path.cwd())
    assert experiments.resolve(None, "Dynamic light scattering").entry_id == "dynamic_light_scattering"
    cases = {
        "Particle yield": "particle_yield",
        "Sandwich-hybridization complex formation": "hybridization_complex_formation",
        "log concentration–log SERS intensity correlation coefficient": "log_concentration_log_sers_intensity_correlation_coefficient",
        "XRD diffraction peak position": "xrd_diffraction_peak_position",
        "Raman peak position": "raman_peak_position",
        "Aspect ratio": "aspect_ratio",
        "Lattice-plane separation": "lattice_plane_spacing",
        "Crystal lattice constant": "crystal_lattice_constant",
    }
    for label, expected_id in cases.items():
        resolved = metrics.resolve(None, label)
        assert resolved is not None, label
        assert resolved.entry_id == expected_id


def _add_edge(g, source, target, relation):
    g.add_edge(source, target, relation=relation)


def test_evidence_topology_diagnostics_detects_missing_chain():
    g = nx.MultiDiGraph()
    g.add_node("m1", type="Measurement", label="gap")
    g.add_node("o1", type="ObservationClaim", label="gap decreases")
    g.add_node("mech1", type="MechanismClaim", label="coupling")
    g.add_node("s1", type="PlasmonicSubstrate", label="substrate")
    rows = evidence_topology_diagnostics(g, domain_profile_id="sers_au_ag")
    codes = {row["code"] for row in rows}
    assert "measurement_without_producer" in codes
    assert "observation_without_support" in codes
    assert "claim_without_application_target" in codes
    assert "mechanism_without_support" in codes


def test_evidence_topology_diagnostics_accepts_complete_chain():
    g = nx.MultiDiGraph()
    g.add_node("e1", type="Experiment")
    g.add_node("m1", type="Measurement")
    g.add_node("o1", type="ObservationClaim")
    g.add_node("mech1", type="MechanismClaim")
    g.add_node("s1", type="PlasmonicSubstrate")
    _add_edge(g, "e1", "m1", "HAS_MEASUREMENT")
    _add_edge(g, "m1", "s1", "MEASURED_FOR")
    _add_edge(g, "m1", "o1", "SUPPORTS_CLAIM")
    _add_edge(g, "o1", "s1", "APPLIES_TO")
    _add_edge(g, "o1", "mech1", "INTERPRETED_AS")
    _add_edge(g, "mech1", "s1", "APPLIES_TO")
    assert evidence_topology_diagnostics(g, domain_profile_id="sers_au_ag") == []


def test_evidence_topology_diagnostics_is_sers_scoped():
    g = nx.MultiDiGraph()
    g.add_node("m1", type="Measurement")
    assert evidence_topology_diagnostics(g, domain_profile_id="dac_her") == []
