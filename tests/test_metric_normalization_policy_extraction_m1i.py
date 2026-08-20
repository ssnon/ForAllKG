from __future__ import annotations

import ast
from pathlib import Path

import pipeline_core.corpus.graph_normalization as normalization
import pipeline_core.corpus.metric_normalization_policy as policy


def test_graph_normalization_reexports_policy_functions():
    assert (
        normalization.refine_distance_metric_id
        is policy.refine_distance_metric_id
    )

    assert (
        normalization.refine_semantic_metric_id
        is policy.refine_semantic_metric_id
    )


def test_chemistry_policy_is_not_owned_by_graph_runtime_source():
    source = Path(
        normalization.__file__
    ).read_text(
        encoding="utf-8"
    )

    forbidden = (
        "exafs-fitted",
        "FT-EXAFS",
        "dft_optimized_bond_length",
        "HAADF",
        "oxidation_state",
        "epr_g_factor",
        "pcohp_antibonding_state_energy",
    )

    for token in forbidden:
        assert token not in source


def test_policy_module_owns_chemistry_refinement_payload():
    source = Path(
        policy.__file__
    ).read_text(
        encoding="utf-8"
    )

    required = (
        "exafs-fitted",
        "dft_optimized_bond_length",
        "haadf",
        "oxidation_state",
        "epr_g_factor",
        "pcohp_antibonding_state_energy",
        "coordination_number",
    )

    for token in required:
        assert token in source


