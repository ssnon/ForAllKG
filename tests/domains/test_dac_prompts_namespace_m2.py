from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.prompts as canonical


ROOT = Path(__file__).resolve().parents[2]






def test_prompt_builder_is_domain_owned():
    assert (
        canonical.build_extraction_prompt.__module__
        == "domains.dac_her.prompts"
    )


def test_canonical_prompt_module_does_not_import_legacy_dac():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "prompts.py"
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
