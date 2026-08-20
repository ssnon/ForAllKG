from __future__ import annotations

import ast
from pathlib import Path

from domains.sers.cross_context_trend import (
    SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER,
)

from dac_her.domains.cross_context_trend_registry import (
    get_cross_context_trend_adapter,
)


ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    modules: list[str] = []

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

    return modules


def test_registry_returns_canonical_adapter() -> None:
    assert (
        get_cross_context_trend_adapter(
            "sers_au_ag"
        )
        is SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
    )


def test_registry_imports_canonical_namespace() -> None:
    path = (
        ROOT
        / "dac_her"
        / "domains"
        / "cross_context_trend_registry.py"
    )

    modules = _imports(path)

    assert (
        "domains.sers.cross_context_trend"
        in modules
    )

    assert (
        "dac_her.domains."
        "sers_au_ag_cross_context_trend"
        not in modules
    )




def test_current_projection_test_imports_canonical_namespace() -> None:
    path = (
        ROOT
        / "tests"
        / "test_sers_cross_context_projection_alpha4c3b.py"
    )

    modules = _imports(path)

    assert (
        "domains.sers.cross_context_trend"
        in modules
    )

    assert (
        "dac_her.domains."
        "sers_au_ag_cross_context_trend"
        not in modules
    )


def test_canonical_module_has_no_legacy_reverse_import() -> None:
    path = (
        ROOT
        / "domains"
        / "sers"
        / "cross_context_trend.py"
    )

    modules = _imports(path)

    assert (
        "dac_her.domains."
        "sers_au_ag_cross_context_trend"
        not in modules
    )
