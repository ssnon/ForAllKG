from __future__ import annotations

import ast
from pathlib import Path

import dac_her.bridge_policy as legacy
import dac_her.domains.dac_her_bridge as adapter_module
import domains.dac_her.bridge_policy as canonical
import pipeline_core.bridge_schemas as bridge_schemas


ROOT = Path(__file__).resolve().parents[1]


POLICY_OBJECTS = (
    "BridgePolicyIssue",
    "BridgePolicyPartition",
    "BridgeRejection",
    "concept_policy_issues",
    "concept_rejection_reasons",
    "filter_bridge_result",
    "partition_bridge_result",
)


def test_legacy_policy_identity_is_preserved():
    for name in POLICY_OBJECTS:
        assert (
            getattr(legacy, name)
            is getattr(canonical, name)
        )


def test_policy_objects_are_domain_owned():
    for name in POLICY_OBJECTS:
        assert (
            getattr(canonical, name).__module__
            == "domains.dac_her.bridge_policy"
        )


def test_policy_version_is_preserved():
    assert (
        legacy.BRIDGE_POLICY_VERSION
        == canonical.BRIDGE_POLICY_VERSION
        == "dac-her-bridge-policy-v2.3.3-calibration"
    )


def test_canonical_policy_uses_core_schema():
    assert (
        canonical.BridgeChunkGraph
        is bridge_schemas.BridgeChunkGraph
    )

    assert (
        canonical.BridgeConcept
        is bridge_schemas.BridgeConcept
    )

    assert (
        canonical.BridgeLink
        is bridge_schemas.BridgeLink
    )


def test_canonical_policy_has_no_legacy_reverse_dependency():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "bridge_policy.py"
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
                or name == "campaigns"
                or name.startswith("campaigns.")
            ):
                violations.append(
                    f"{path}:{node.lineno}:{name}"
                )

    assert violations == []


def test_adapter_binds_canonical_policy_objects():
    adapter = (
        adapter_module.DAC_HER_BRIDGE_ADAPTER
    )

    assert (
        adapter.partition_result
        is canonical.partition_bridge_result
    )

    assert (
        adapter.policy_version
        == canonical.BRIDGE_POLICY_VERSION
    )


def test_adapter_policy_provenance_is_canonical():
    adapter = (
        adapter_module.DAC_HER_BRIDGE_ADAPTER
    )

    canonical_path = Path(
        canonical.__file__
    ).resolve()

    legacy_path = Path(
        legacy.__file__
    ).resolve()

    paths = {
        Path(value).resolve()
        for value
        in adapter.implementation_files.policy
    }

    assert canonical_path in paths
    assert legacy_path not in paths


def test_refilter_imports_canonical_policy_module():
    path = (
        ROOT
        / "scripts"
        / "refilter_bridge_graph.py"
    )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    imports = []
    froms = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            froms.append(
                node.module or ""
            )

    assert (
        "domains.dac_her.bridge_policy"
        in imports
    )

    assert (
        "dac_her.bridge_policy"
        not in imports
    )

    # M3 moves the canonical policy-run implementation
    # into pipeline_core while retaining the DAC facade
    # as the runtime compatibility entrypoint.
    assert (
        "pipeline_core.bridge_policy_run"
        in imports
    )

    assert (
        "dac_her.bridge_policy_run"
        not in imports
    )

    assert (
        "dac_her.bridge_policy_run"
        in froms
    )
