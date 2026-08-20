from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.bridge_recovery_prompts as canonical

import domains.dac_her.bridge as bridge_adapter_module


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_VERSION = (
    "dac-her-bridge-candidate-recovery-v1"
)






def test_builder_is_domain_owned():
    assert (
        canonical
        .build_bridge_candidate_repair_prompt
        .__module__
        == "domains.dac_her.bridge_recovery_prompts"
    )


def test_adapter_uses_canonical_recovery_semantics():
    adapter = (
        bridge_adapter_module
        .DAC_HER_BRIDGE_ADAPTER
    )

    assert (
        adapter.recovery_prompt_version
        == canonical.BRIDGE_RECOVERY_PROMPT_VERSION
    )

    assert (
        adapter.recovery_system_prompt
        == canonical.BRIDGE_RECOVERY_SYSTEM_PROMPT
    )

    assert (
        adapter.build_candidate_repair_prompt
        is canonical.build_bridge_candidate_repair_prompt
    )


def test_adapter_source_imports_canonical_recovery_module():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "bridge.py"
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
        "domains.dac_her.bridge_recovery_prompts"
        in modules
    )


def test_canonical_recovery_prompt_has_no_legacy_import():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "bridge_recovery_prompts.py"
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
