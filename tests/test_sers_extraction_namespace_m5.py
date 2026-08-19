from __future__ import annotations

import ast
from pathlib import Path

from domains.sers.extraction import (
    SERS_AU_AG_EXTRACTION_ADAPTER as CANONICAL_ADAPTER,
)
from dac_her.domains.extraction_registry import (
    get_extraction_adapter,
)
from dac_her.domains.sers_au_ag_extraction import (
    SERS_AU_AG_EXTRACTION_ADAPTER as LEGACY_ADAPTER,
)


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_adapter_is_canonical() -> None:
    assert LEGACY_ADAPTER is CANONICAL_ADAPTER


def test_registry_returns_canonical_adapter() -> None:
    assert (
        get_extraction_adapter(
            "sers_au_ag"
        )
        is CANONICAL_ADAPTER
    )


def test_registry_imports_canonical_namespace() -> None:
    path = (
        ROOT
        / "dac_her"
        / "domains"
        / "extraction_registry.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    modules = [
        node.module or ""
        for node in tree.body
        if isinstance(
            node,
            ast.ImportFrom,
        )
    ]

    assert (
        "domains.sers.extraction"
        in modules
    )

    assert (
        "dac_her.domains.sers_au_ag_extraction"
        not in modules
    )


def test_legacy_module_is_definition_free() -> None:
    path = (
        ROOT
        / "dac_her"
        / "domains"
        / "sers_au_ag_extraction.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
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


def test_canonical_extraction_uses_canonical_prompts() -> None:
    path = (
        ROOT
        / "domains"
        / "sers"
        / "extraction.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    modules = [
        node.module or ""
        for node in tree.body
        if isinstance(
            node,
            ast.ImportFrom,
        )
    ]

    assert (
        "domains.sers.prompts"
        in modules
    )

    assert (
        "dac_her.sers_prompts"
        not in modules
    )


def test_canonical_extraction_has_no_legacy_reverse_import() -> None:
    path = (
        ROOT
        / "domains"
        / "sers"
        / "extraction.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(
                node.module or ""
            )

        elif isinstance(node, ast.Import):
            modules.extend(
                alias.name
                for alias in node.names
            )

    assert not any(
        module.startswith(
            "dac_her.domains.sers_au_ag"
        )
        for module in modules
    )
