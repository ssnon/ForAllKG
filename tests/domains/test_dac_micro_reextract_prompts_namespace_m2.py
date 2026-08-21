from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.micro_reextract_prompts as canonical
from domains.extraction_registry import get_extraction_adapter

from pipeline_core.runtime.validation_issues import (
    IssueCode,
    ValidationReport,
)


ROOT = Path(__file__).resolve().parents[2]








def test_public_builders_are_domain_owned():
    assert (
        canonical.build_micro_reextract_prompt.__module__
        == "domains.dac_her.micro_reextract_prompts"
    )

    assert (
        canonical.build_domain_gate_recovery_prompt.__module__
        == "domains.dac_her.micro_reextract_prompts"
    )


def test_canonical_module_has_no_legacy_dac_import():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "micro_reextract_prompts.py"
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
            ):
                violations.append(
                    f"{path}:{node.lineno}:{name}"
                )

    assert violations == []



def test_extract_paper_routes_prompt_provenance_through_adapter():
    path = ROOT / "scripts" / "corpus" / "extract_paper.py"

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Attribute):
            continue

        if (
            node.func.attr
            == "prompt_builder_implementation_paths"
        ):
            calls.append(node.lineno)

    assert len(calls) == 1



def test_dac_adapter_provenance_includes_micro_prompt_file():
    adapter = get_extraction_adapter("dac_her")

    paths = {
        Path(path).resolve()
        for path
        in adapter.prompt_builder_implementation_paths()
    }

    assert (
        Path(canonical.__file__).resolve()
        in paths
    )
