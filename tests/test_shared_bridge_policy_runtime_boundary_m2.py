from __future__ import annotations

import ast
from pathlib import Path

import dac_her.bridge_policy_runtime as legacy
import domains.sers.bridge as sers_adapter_module
import pipeline_core.bridge_policy_runtime as core


ROOT = Path(__file__).resolve().parents[1]


PUBLIC_RUNTIME = (
    "PluginBridgePolicyIssue",
    "PluginBridgeRejection",
    "PluginBridgePolicyPartition",
    "dedupe_policy_issues",
    "partition_with_policy",
)


def test_legacy_public_runtime_identity_is_preserved():
    for name in PUBLIC_RUNTIME:
        assert (
            getattr(legacy, name)
            is getattr(core, name)
        )


def test_runtime_is_core_owned():
    for name in PUBLIC_RUNTIME:
        assert (
            getattr(core, name).__module__
            == "pipeline_core.bridge_policy_runtime"
        )


def test_core_has_no_reverse_dependency():
    path = (
        ROOT
        / "pipeline_core"
        / "bridge_policy_runtime.py"
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
                or name == "domains"
                or name.startswith("domains.")
                or name == "campaigns"
                or name.startswith("campaigns.")
            ):
                violations.append(
                    f"{path}:{node.lineno}:{name}"
                )

    assert violations == []


def test_sers_policy_imports_canonical_runtime():
    path = (
        ROOT / "domains" / "sers" / "bridge_policy.py"
    )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    modules = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]

    assert (
        "pipeline_core.bridge_policy_runtime"
        in modules
    )

    assert (
        "dac_her.bridge_policy_runtime"
        not in modules
    )


def test_sers_adapter_provenance_uses_core_runtime():
    adapter = (
        sers_adapter_module.SERS_AU_AG_BRIDGE_ADAPTER
    )

    core_path = Path(
        core.__file__
    ).resolve()

    legacy_path = Path(
        legacy.__file__
    ).resolve()

    policy = {
        Path(value).resolve()
        for value
        in adapter.implementation_files.policy
    }

    assert core_path in policy
    assert legacy_path not in policy
