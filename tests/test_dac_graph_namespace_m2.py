from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.graph as canonical

from domains.graph_registry import (
    get_graph_adapter,
)
from domains.dac_her.semantic_roles import (
    normalize_measurement_subject_roles,
)


ROOT = Path(__file__).resolve().parents[1]




def test_registry_resolves_canonical_adapter():
    assert (
        get_graph_adapter("dac_her")
        is canonical.DAC_HER_GRAPH_ADAPTER
    )


def test_adapter_identity_semantics_are_preserved():
    adapter = canonical.DAC_HER_GRAPH_ADAPTER

    assert adapter.adapter_id == "dac_her"
    assert adapter.domain_profile_id == "dac_her"
    assert adapter.relation_constraints == ()

    assert (
        adapter.semantic_role_normalizer
        is normalize_measurement_subject_roles
    )


def test_adapter_uses_shared_contract_type():
    assert (
        type(
            canonical.DAC_HER_GRAPH_ADAPTER
        ).__module__
        == "pipeline_core.corpus.graph.graph_domain"
    )


def test_canonical_graph_module_does_not_import_legacy_dac():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "graph.py"
    )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    violations = []

    for node in ast.walk(tree):
        names = []

        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""]

        elif isinstance(node, ast.Import):
            names = [
                alias.name
                for alias in node.names
            ]

        for name in names:
            if (
                name == "dac_her"
                or name.startswith("dac_her.")
            ):
                violations.append(
                    f"{path}:{node.lineno}:{name}"
                )

    assert violations == []
