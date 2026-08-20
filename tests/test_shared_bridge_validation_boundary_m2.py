from __future__ import annotations

import ast
import inspect
from pathlib import Path

import domains.dac_her.bridge_validation as legacy
import dac_her.domains.dac_her_bridge as dac_adapter_module
import domains.sers.bridge as sers_adapter_module
import domains.dac_her.scientific_signatures as dac_signatures
import pipeline_core.bridge_schemas as bridge_schemas
import pipeline_core.bridge_validation as core


ROOT = Path(__file__).resolve().parents[1]


def test_core_requires_explicit_domain_hook():
    parameter = inspect.signature(
        core.bridge_validation_issues
    ).parameters[
        "anchor_context_issues_fn"
    ]

    assert parameter.default is inspect.Parameter.empty


def test_dac_binding_preserves_default_hook():
    parameter = inspect.signature(
        legacy.bridge_validation_issues
    ).parameters[
        "anchor_context_issues_fn"
    ]

    assert (
        parameter.default
        is dac_signatures.strong_anchor_context_issues
    )


def test_dac_binding_public_function_ownership_is_canonical():
    assert (
        legacy.bridge_validation_issues.__module__
        == "domains.dac_her.bridge_validation"
    )

    assert (
        legacy.validate_bridge_chunk.__module__
        == "domains.dac_her.bridge_validation"
    )

    assert (
        legacy.bind_bridge_validation.__module__
        == "domains.dac_her.bridge_validation"
    )


def test_dac_binding_binder_preserves_nested_function_module():
    def hook(**kwargs):
        del kwargs
        return []

    issues, validate = (
        legacy.bind_bridge_validation(hook)
    )

    assert (
        issues.__module__
        == "domains.dac_her.bridge_validation"
    )

    assert (
        validate.__module__
        == "domains.dac_her.bridge_validation"
    )


def test_dac_adapter_uses_canonical_callbacks_and_tracks_core():
    adapter = (
        dac_adapter_module.DAC_HER_BRIDGE_ADAPTER
    )

    assert (
        adapter.validation_issues
        is legacy.bridge_validation_issues
    )

    assert (
        adapter.validate_chunk
        is legacy.validate_bridge_chunk
    )

    paths = {
        Path(value).resolve()
        for value
        in adapter.implementation_files.extraction
    }

    assert Path(
        legacy.__file__
    ).resolve() in paths

    assert Path(
        core.__file__
    ).resolve() in paths


def test_sers_adapter_binds_directly_to_core():
    adapter = (
        sers_adapter_module.SERS_AU_AG_BRIDGE_ADAPTER
    )

    assert (
        adapter.validation_issues.__module__
        == "pipeline_core.bridge_validation"
    )

    assert (
        adapter.validate_chunk.__module__
        == "pipeline_core.bridge_validation"
    )

    paths = {
        Path(value).resolve()
        for value
        in adapter.implementation_files.extraction
    }

    assert Path(
        core.__file__
    ).resolve() in paths

    assert Path(
        legacy.__file__
    ).resolve() not in paths


def test_core_uses_canonical_bridge_schema():
    assert (
        core.BridgeChunkGraph
        is bridge_schemas.BridgeChunkGraph
    )


def test_core_has_no_reverse_domain_dependency():
    path = (
        ROOT
        / "pipeline_core"
        / "bridge_validation.py"
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
