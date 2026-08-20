from __future__ import annotations

import ast
from pathlib import Path

import domains.sers.bridge as bridge_module

from domains.sers.bridge import (
    SERS_AU_AG_BRIDGE_ADAPTER,
)

from domains.bridge_registry import (
    get_bridge_adapter,
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
        if isinstance(node, ast.ImportFrom):
            modules.append(
                node.module or ""
            )

        elif isinstance(node, ast.Import):
            modules.extend(
                alias.name
                for alias in node.names
            )

    return modules


def test_registry_returns_canonical_bridge_adapter() -> None:
    assert (
        get_bridge_adapter(
            "sers_au_ag"
        )
        is SERS_AU_AG_BRIDGE_ADAPTER
    )


def test_registry_imports_canonical_bridge() -> None:
    path = (
        ROOT
        / "domains"
        / "bridge_registry.py"
    )

    modules = _imports(path)

    assert "domains.sers.bridge" in modules

    assert (
        "dac_her.domains.sers_au_ag_bridge"
        not in modules
    )


def test_canonical_bridge_uses_canonical_helpers() -> None:
    path = (
        ROOT
        / "domains"
        / "sers"
        / "bridge.py"
    )

    modules = _imports(path)

    expected = {
        "domains.sers.bridge_policy",
        "domains.sers.bridge_prompts",
        "domains.sers.bridge_recovery_prompts",
        "domains.sers.bridge_signatures",
    }

    assert expected.issubset(
        set(modules)
    )


def test_canonical_policy_uses_canonical_signatures() -> None:
    path = (
        ROOT
        / "domains"
        / "sers"
        / "bridge_policy.py"
    )

    modules = _imports(path)

    assert (
        "domains.sers.bridge_signatures"
        in modules
    )

    assert (
        "dac_her.sers_bridge_signatures"
        not in modules
    )


def test_canonical_bridge_has_no_legacy_cluster_import() -> None:
    modules = _imports(
        ROOT
        / "domains"
        / "sers"
        / "bridge.py"
    )

    forbidden = (
        "dac_her.domains.sers_au_ag_bridge",
        "dac_her.sers_bridge_policy",
        "dac_her.sers_bridge_prompts",
        "dac_her.sers_bridge_recovery_prompts",
        "dac_her.sers_bridge_signatures",
    )

    assert not any(
        module.startswith(forbidden)
        for module in modules
    )


def test_module_object_is_canonical() -> None:
    assert (
        bridge_module.SERS_AU_AG_BRIDGE_ADAPTER
        is SERS_AU_AG_BRIDGE_ADAPTER
    )
