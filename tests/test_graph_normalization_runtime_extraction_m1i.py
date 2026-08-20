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
