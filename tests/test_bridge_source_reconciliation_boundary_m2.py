from __future__ import annotations

import ast
from pathlib import Path

import pipeline_core.corpus.bridge.bridge_source_reconciliation as core


ROOT = Path(__file__).resolve().parents[1]


PUBLIC_SYMBOLS = (
    "ReconciledSpan",
    "SourceNormalizationOperation",
    "normalized_scientific_text",
    "reconcile_phrase_to_text",
    "reconcile_concept_payload",
)

PRIVATE_COMPAT_SYMBOLS = (
    "_visible_markdown_text_with_map",
    "_drop_html_tags_with_map",
    "_normalized_with_map",
    "_unique_occurrence",
)








def test_functions_are_core_owned():
    assert (
        core.reconcile_phrase_to_text.__module__
        == "pipeline_core.corpus.bridge.bridge_source_reconciliation"
    )

    assert (
        core.reconcile_concept_payload.__module__
        == "pipeline_core.corpus.bridge.bridge_source_reconciliation"
    )


def test_bridge_recovery_imports_core_reconciliation():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "bridge_recovery.py"
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
        "pipeline_core.corpus.bridge.bridge_source_reconciliation"
        in modules
    )


def test_core_module_has_no_domain_reverse_dependency():
    path = (
        ROOT
        / "pipeline_core"
        / "corpus/bridge/bridge_source_reconciliation.py"
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
