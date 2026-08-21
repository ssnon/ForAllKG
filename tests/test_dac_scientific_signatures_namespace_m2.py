from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.scientific_signatures as canonical

import domains.dac_her.bridge as adapter_module


ROOT = Path(__file__).resolve().parents[1]


PUBLIC_FUNCTIONS = (
    "normalize_scientific_text",
    "nuclearity_signature",
    "support_signature",
    "model_physical_signature",
    "node_scientific_signature",
    "strict_node_catalog",
    "strong_anchor_context_issues",
)




def test_functions_are_domain_owned():
    for name in PUBLIC_FUNCTIONS:
        assert (
            getattr(canonical, name).__module__
            == "domains.dac_her.scientific_signatures"
        )


def test_shared_chemistry_dependency_is_core_owned():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "scientific_signatures.py"
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
        "pipeline_core.corpus.extraction.chemistry_signatures"
        in modules
    )

    assert not any(
        name == "dac_her"
        or name.startswith("dac_her.")
        or name == "campaigns"
        or name.startswith("campaigns.")
        for name in modules
    )


def test_adapter_binds_canonical_signature_functions():
    adapter = adapter_module.DAC_HER_BRIDGE_ADAPTER

    assert (
        adapter.strict_node_catalog_builder
        is canonical.strict_node_catalog
    )

    assert (
        adapter.anchor_context_issues
        is canonical.strong_anchor_context_issues
    )


def test_adapter_provenance_uses_canonical_source():
    adapter = adapter_module.DAC_HER_BRIDGE_ADAPTER

    canonical_path = Path(
        canonical.__file__
    ).resolve()

    legacy_path = (
        ROOT
        / "dac_her"
        / "scientific_signatures.py"
    ).resolve()

    extraction = {
        Path(value).resolve()
        for value
        in adapter.implementation_files.extraction
    }

    policy = {
        Path(value).resolve()
        for value
        in adapter.implementation_files.policy
    }

    assert canonical_path in extraction
    assert canonical_path in policy

    assert legacy_path not in extraction
    assert legacy_path not in policy


def test_bridge_validation_and_policy_import_canonical():
    for path in (
        ROOT / "domains" / "dac_her" / "bridge_validation.py",
        ROOT / "domains" / "dac_her" / "bridge_policy.py",
    ):
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
            "domains.dac_her.scientific_signatures"
            in modules
        )

        assert (
            "dac_her.scientific_signatures"
            not in modules
        )




def test_canonical_module_has_no_reverse_dependency():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "scientific_signatures.py"
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
