from __future__ import annotations

import ast
from pathlib import Path

from domains.sers.graph import (
    SERS_AU_AG_GRAPH_ADAPTER as CANONICAL_ADAPTER,
    SERS_RELATION_CONSTRAINTS as CANONICAL_CONSTRAINTS,
)
from dac_her.domains.graph_registry import (
    get_graph_adapter,
)
from dac_her.domains.sers_au_ag_graph import (
    SERS_AU_AG_GRAPH_ADAPTER as LEGACY_ADAPTER,
    SERS_RELATION_CONSTRAINTS as LEGACY_CONSTRAINTS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_adapter_is_canonical() -> None:
    assert LEGACY_ADAPTER is CANONICAL_ADAPTER


def test_legacy_constraints_are_canonical() -> None:
    assert LEGACY_CONSTRAINTS is CANONICAL_CONSTRAINTS


def test_registry_returns_canonical_adapter() -> None:
    assert (
        get_graph_adapter("sers_au_ag")
        is CANONICAL_ADAPTER
    )


def test_graph_registry_imports_canonical_namespace() -> None:
    path = (
        ROOT
        / "dac_her"
        / "domains"
        / "graph_registry.py"
    )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    modules = [
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ]

    assert "domains.sers.graph" in modules
    assert (
        "dac_her.domains.sers_au_ag_graph"
        not in modules
    )


def test_legacy_graph_module_is_definition_free() -> None:
    path = (
        ROOT
        / "dac_her"
        / "domains"
        / "sers_au_ag_graph.py"
    )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    definitions = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
    ]

    assert definitions == []


def test_canonical_graph_has_no_legacy_reverse_import() -> None:
    path = (
        ROOT
        / "domains"
        / "sers"
        / "graph.py"
    )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")

        elif isinstance(node, ast.Import):
            modules.extend(
                alias.name
                for alias in node.names
            )

    assert (
        "dac_her.domains.sers_au_ag_graph"
        not in modules
    )
