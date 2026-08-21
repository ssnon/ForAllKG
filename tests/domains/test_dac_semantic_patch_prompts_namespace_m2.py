from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.semantic_patch_prompts as canonical

from pipeline_core.runtime.validation_issues import (
    ValidationReport,
)


ROOT = Path(__file__).resolve().parents[2]








def test_canonical_functions_are_domain_owned():
    assert (
        canonical.build_semantic_patch_prompt.__module__
        == "domains.dac_her.semantic_patch_prompts"
    )

    assert (
        canonical.build_patch_rejection_feedback.__module__
        == "domains.dac_her.semantic_patch_prompts"
    )


def test_canonical_module_has_no_legacy_dac_import():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "semantic_patch_prompts.py"
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


def test_extract_paper_provenance_imports_canonical_module():
    path = ROOT / "scripts" / "corpus/extract_paper.py"

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    bindings = []

    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue

        for alias in node.names:
            if (
                alias.asname
                == "semantic_patch_prompts_module"
            ):
                bindings.append(alias.name)

    assert bindings == [
        "domains.dac_her.semantic_patch_prompts"
    ]


def test_extract_paper_still_hashes_prompt_module_file():
    path = ROOT / "scripts" / "corpus/extract_paper.py"

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    refs = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue

        if node.attr != "__file__":
            continue

        if (
            isinstance(node.value, ast.Name)
            and node.value.id
            == "semantic_patch_prompts_module"
        ):
            refs.append(node.lineno)

    assert len(refs) == 1
