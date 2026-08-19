from __future__ import annotations

import ast
from pathlib import Path

from domains.sers.profile import (
    SERS_AU_AG_PROFILE as CANONICAL_PROFILE,
)
from dac_her.domains.registry import (
    get_domain_profile,
)
from dac_her.domains.sers_au_ag import (
    SERS_AU_AG_PROFILE as LEGACY_PROFILE,
)


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_profile_is_legacy_profile() -> None:
    assert (
        CANONICAL_PROFILE
        is LEGACY_PROFILE
    )


def test_registry_returns_canonical_profile() -> None:
    assert (
        get_domain_profile(
            "sers_au_ag"
        )
        is CANONICAL_PROFILE
    )


def test_registry_imports_canonical_profile_namespace() -> None:
    path = (
        ROOT
        / "dac_her"
        / "domains"
        / "registry.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    imports = [
        node.module or ""
        for node in tree.body
        if isinstance(
            node,
            ast.ImportFrom,
        )
    ]

    assert (
        "domains.sers.profile"
        in imports
    )

    assert (
        "dac_her.domains.sers_au_ag"
        not in imports
    )


def test_legacy_profile_module_is_definition_free() -> None:
    path = (
        ROOT
        / "dac_her"
        / "domains"
        / "sers_au_ag.py"
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


def test_canonical_profile_does_not_reverse_import_legacy() -> None:
    path = (
        ROOT
        / "domains"
        / "sers"
        / "profile.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    modules = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.ImportFrom,
        ):
            modules.append(
                node.module or ""
            )

        elif isinstance(
            node,
            ast.Import,
        ):
            modules.extend(
                alias.name
                for alias in node.names
            )

    assert (
        "dac_her.domains.sers_au_ag"
        not in modules
    )
