from __future__ import annotations

import ast
from pathlib import Path

from domains.sers.trend_alpha4c2121 import (
    SERS_AU_AG_TREND_ADAPTER as CANONICAL_TREND,
)
from domains.sers.trend_precision_alpha4c21211 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER as CANONICAL_PRECISION,
)

from dac_her.domains.trend_registry import (
    get_trend_adapter,
)
from dac_her.domains.trend_precision_registry import (
    get_trend_precision_adapter,
)


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_MODULES = (
    "trend.py",
    "trend_alpha4c21.py",
    "trend_alpha4c211.py",
    "trend_alpha4c212.py",
    "trend_alpha4c2121.py",
    "trend_alpha4c5g2.py",
    "trend_alpha4c5g2r1.py",
    "trend_alpha4c5g2r2.py",
    "trend_precision.py",
    "trend_precision_alpha4c211.py",
    "trend_precision_alpha4c212.py",
    "trend_precision_alpha4c2121.py",
    "trend_precision_alpha4c21211.py",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    result: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            result.append(node.module or "")

        elif isinstance(node, ast.Import):
            result.extend(
                alias.name
                for alias in node.names
            )

    return result


def test_current_registries_return_canonical_adapters() -> None:
    assert (
        get_trend_adapter("sers_au_ag")
        is CANONICAL_TREND
    )

    assert (
        get_trend_precision_adapter("sers_au_ag")
        is CANONICAL_PRECISION
    )


def test_trend_registry_imports_canonical_current_provider() -> None:
    modules = _imports(
        ROOT
        / "dac_her"
        / "domains"
        / "trend_registry.py"
    )

    assert (
        "domains.sers.trend_alpha4c2121"
        in modules
    )

    assert (
        "dac_her.domains."
        "sers_au_ag_trend_alpha4c2121"
        not in modules
    )


def test_precision_registry_imports_canonical_current_provider() -> None:
    modules = _imports(
        ROOT
        / "dac_her"
        / "domains"
        / "trend_precision_registry.py"
    )

    assert (
        "domains.sers."
        "trend_precision_alpha4c21211"
        in modules
    )

    assert (
        "dac_her.domains."
        "sers_au_ag_trend_precision_alpha4c21211"
        not in modules
    )


def test_all_canonical_lineage_modules_exist() -> None:
    root = (
        ROOT
        / "domains"
        / "sers"
    )

    for filename in CANONICAL_MODULES:
        assert (
            root / filename
        ).is_file()


def test_canonical_lineage_has_no_legacy_lineage_imports() -> None:
    root = (
        ROOT
        / "domains"
        / "sers"
    )

    violations = []

    for filename in CANONICAL_MODULES:
        path = root / filename

        for module in _imports(path):
            if module.startswith(
                "dac_her.domains."
                "sers_au_ag_trend"
            ):
                violations.append(
                    (
                        filename,
                        module,
                    )
                )

    assert violations == []


def test_canonical_internal_edges_stay_in_domain_namespace() -> None:
    root = (
        ROOT
        / "domains"
        / "sers"
    )

    internal = []

    for filename in CANONICAL_MODULES:
        path = root / filename

        for module in _imports(path):
            if module.startswith(
                "domains.sers.trend"
            ):
                internal.append(
                    (
                        filename,
                        module,
                    )
                )

    assert len(internal) == 12
