from __future__ import annotations

import ast
from pathlib import Path

import pipeline_core.corpus.graph_normalization as facade
import pipeline_core.graph_normalization_runtime as runtime


def test_vocabulary_issue_is_reexported_from_core_runtime():
    assert (
        facade.VocabularyIssue
        is runtime.VocabularyIssue
    )


def test_core_runtime_has_no_dac_her_dependency():
    source = Path(
        runtime.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert "dac_her" not in source


def test_facade_owns_historical_scientific_bindings():
    source = Path(
        facade.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "metric_normalization_policy"
        in source
    )

    assert (
        "PARAMETER_CONDITION_NAMES"
        in source
    )

    assert (
        "_normalize_graph_vocabularies"
        in source
    )

    assert (
        "_normalize_networkx_metric_vocabularies"
        in source
    )


def test_extraction_provenance_tracks_facade_core_and_policy():
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

    calls = [
        node
        for node in ast.walk(main)
        if (
            isinstance(node, ast.Call)
            and isinstance(
                node.func,
                ast.Name,
            )
            and node.func.id
            == "compute_run_metadata"
        )
    ]

    assert len(calls) == 1

    keyword = next(
        item
        for item in calls[0].keywords
        if item.arg
        == "implementation_paths"
    )

    rendered = ast.unparse(
        keyword.value
    )

    assert (
        "graph_normalization_module.__file__"
        in rendered
    )

    assert (
        "graph_normalization_runtime_module.__file__"
        in rendered
    )

    assert (
        "metric_normalization_policy_module.__file__"
        in rendered
    )
