from __future__ import annotations

from pathlib import Path

from domains.graph_registry import get_graph_adapter
from domains.sers.graph_diagnostics import (
    collect_sers_graph_diagnostics,
)


def test_generic_graph_semantics_has_no_sers_policy_literals():
    source = Path(
        "pipeline_core/corpus/graph_semantics.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "sers_au_ag",
        "SERS_GRAPH_DIAGNOSTICS_VERSION",
        "PlasmonicSubstrate",
        "RamanReporter",
        "normal raman",
        "without sers substrate",
    )

    for literal in forbidden:
        assert literal not in source


def test_sers_graph_adapter_owns_diagnostics_collector():
    adapter = get_graph_adapter("sers_au_ag")

    assert (
        adapter.diagnostics_collector
        is collect_sers_graph_diagnostics
    )
    assert adapter.primary_subject_types == frozenset({
        "PlasmonicSubstrate",
        "Nanostructure",
    })
