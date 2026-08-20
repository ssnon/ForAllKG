from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

import pipeline_core.corpus.graph_normalization as normalization


def test_chemistry_metric_refinement_contract():
    cases = (
        (
            "bond_length",
            "Bond length",
            ("quantitative EXAFS fitting",),
            "fitted_scattering_path_length",
        ),
        (
            "bond_distance",
            "Bond distance",
            ("FT-EXAFS peak position",),
            "exafs_radial_peak_position",
        ),
        (
            "bond_length",
            "Bond length",
            ("DFT optimized geometry",),
            "dft_optimized_bond_length",
        ),
        (
            "bond_distance",
            "Bond distance",
            ("HAADF-STEM directly imaged atomic pairs",),
            "interatomic_distance",
        ),
        (
            "other",
            "Average oxidation state",
            ("valence state analysis",),
            "oxidation_state",
        ),
        (
            "other",
            "EPR g value",
            ("EPR g = 2.03",),
            "epr_g_factor",
        ),
        (
            "other",
            "pCOHP antibonding state energy",
            ("pCOHP antibond energy",),
            "pcohp_antibonding_state_energy",
        ),
        (
            "other",
            "Coordination number",
            ("CN = 4",),
            "coordination_number",
        ),
    )

    for (
        entry_id,
        label,
        source_texts,
        expected,
    ) in cases:
        assert (
            normalization.refine_semantic_metric_id(
                entry_id=entry_id,
                label=label,
                source_texts=source_texts,
            )
            == expected
        )


def test_neutral_metric_refinement_is_identity():
    assert (
        normalization.refine_semantic_metric_id(
            entry_id="sers_enhancement_factor",
            label="SERS enhancement factor",
            source_texts=(
                "enhancement factor measured for analyte",
            ),
        )
        == "sers_enhancement_factor"
    )


def test_graph_normalizer_uses_shared_metric_refinement_seam(
    monkeypatch,
):
    calls = []

    def fake_refiner(
        *,
        entry_id,
        label,
        source_texts,
    ):
        calls.append(
            (
                entry_id,
                label,
                tuple(source_texts),
            )
        )
        return entry_id

    monkeypatch.setattr(
        normalization,
        "refine_semantic_metric_id",
        fake_refiner,
    )

    source = Path(
        normalization.__file__
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    function = next(
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name
            == "normalize_graph_vocabularies"
        )
    )

    call_names = {
        node.func.id
        for node in ast.walk(function)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        )
    }

    assert (
        "refine_semantic_metric_id"
        in call_names
    )


def test_networkx_normalizer_uses_same_metric_refinement_seam():
    source = Path(
        normalization.__file__
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    function = next(
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name
            == "normalize_networkx_metric_vocabularies"
        )
    )

    call_names = {
        node.func.id
        for node in ast.walk(function)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        )
    }

    assert (
        "refine_semantic_metric_id"
        in call_names
    )


def test_extraction_run_fingerprint_tracks_graph_normalization_source():
    source = Path(
        "scripts/extract_paper.py"
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    main = next(
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "main"
        )
    )

    compute_calls = [
        node
        for node in ast.walk(main)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "compute_run_metadata"
        )
    ]

    assert len(compute_calls) == 1

    call = compute_calls[0]

    implementation_keyword = next(
        keyword
        for keyword in call.keywords
        if keyword.arg
        == "implementation_paths"
    )

    rendered = ast.unparse(
        implementation_keyword.value
    )

    assert (
        "graph_normalization_module.__file__"
        in rendered
    )


def test_run_metadata_hashes_each_implementation_file():
    source = Path(
        "dac_her/run_state.py"
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    function = next(
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name
            == "compute_run_metadata"
        )
    )

    string_constants = {
        node.value
        for node in ast.walk(function)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )
    }

    call_names = {
        node.func.id
        for node in ast.walk(function)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        )
    }

    assert "implementation_files" in string_constants
    assert "run_fingerprint" in string_constants
    assert "run_id" in string_constants

    assert "sha256_file" in call_names

    rendered = ast.unparse(function)

    assert (
        "for path in sorted"
        in rendered
    )
    assert (
        "implementation_paths"
        in rendered
    )
