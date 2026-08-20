from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.bridge_prompts as canonical

import dac_her.domains.dac_her_bridge as bridge_adapter_module


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_VERSION = (
    "dac-her-bridge-v2.3.1-calibration"
)






def test_prompt_builders_are_domain_owned():
    assert (
        canonical.build_bridge_prompt.__module__
        == "domains.dac_her.bridge_prompts"
    )

    assert (
        canonical.build_bridge_repair_prompt.__module__
        == "domains.dac_her.bridge_prompts"
    )


def test_adapter_prompt_semantics_use_canonical_values():
    adapter = (
        bridge_adapter_module
        .DAC_HER_BRIDGE_ADAPTER
    )

    assert (
        adapter.prompt_version
        == canonical.BRIDGE_PROMPT_VERSION
    )

    assert (
        adapter.system_prompt
        == canonical.BRIDGE_SYSTEM_PROMPT
    )


def test_adapter_implementation_files_bind_canonical_prompt():
    adapter = (
        bridge_adapter_module
        .DAC_HER_BRIDGE_ADAPTER
    )

    canonical_path = Path(
        canonical.__file__
    ).resolve()

    files = {
        Path(value).resolve()
        for value
        in adapter.implementation_files.extraction
    }

    assert canonical_path in files


def test_adapter_source_imports_canonical_prompt_module():
    path = (
        ROOT
        / "dac_her"
        / "domains"
        / "dac_her_bridge.py"
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
        "domains.dac_her.bridge_prompts"
        in modules
    )


def test_canonical_prompt_has_no_legacy_import():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "bridge_prompts.py"
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
