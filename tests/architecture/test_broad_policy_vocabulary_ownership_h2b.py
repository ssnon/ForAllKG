from __future__ import annotations

import ast
from pathlib import Path

from domains.catalysis_mechanism.extraction_policy import (
    BROAD_ABSTRACT_RECOVERY_POLICY_ID,
    broad_abstract_extraction_policy,
)
from domains.catalysis_mechanism.vocabulary_context import (
    BROAD_METHODS_ONLY_CONTEXT_ID,
    build_broad_experiment_methods_vocabulary_context,
)
from domains.extraction_registry import get_extraction_adapter


ROOT = Path(__file__).resolve().parents[2]


def test_broad_policy_and_vocabulary_are_domain_owned() -> None:
    assert not (
        ROOT
        / "pipeline_core"
        / "corpus"
        / "broad_extraction_policy.py"
    ).exists()

    assert not (
        ROOT
        / "pipeline_core"
        / "corpus"
        / "extraction_vocabulary_context.py"
    ).exists()

    policy_path = (
        ROOT
        / "domains"
        / "catalysis_mechanism"
        / "extraction_policy.py"
    )

    vocabulary_path = (
        ROOT
        / "domains"
        / "catalysis_mechanism"
        / "vocabulary_context.py"
    )

    assert policy_path.exists()
    assert vocabulary_path.exists()

    adapter = get_extraction_adapter(
        "catalysis_mechanism"
    )

    assert (
        adapter.extraction_policy_id
        == BROAD_ABSTRACT_RECOVERY_POLICY_ID
    )

    assert (
        adapter.extraction_policy_transform
        is broad_abstract_extraction_policy
    )

    assert (
        adapter.reduced_vocabulary_context_id
        == BROAD_METHODS_ONLY_CONTEXT_ID
    )

    assert (
        adapter.reduced_vocabulary_context_builder
        is build_broad_experiment_methods_vocabulary_context
    )

    assert {
        Path(path).resolve()
        for path
        in adapter.extraction_policy_implementation_paths()
    } == {
        policy_path.resolve()
    }

    assert {
        Path(path).resolve()
        for path
        in adapter.reduced_vocabulary_context_implementation_paths()
    } == {
        vocabulary_path.resolve()
    }


def test_extract_paper_uses_adapter_capabilities_without_domain_imports() -> None:
    path = (
        ROOT
        / "scripts"
        / "corpus"
        / "extract_paper.py"
    )

    text = path.read_text(
        encoding="utf-8"
    )

    assert (
        "broad_abstract_extraction_policy"
        not in text
    )

    assert (
        "BROAD_ABSTRACT_RECOVERY_POLICY_ID"
        not in text
    )

    assert (
        "build_broad_experiment_methods_vocabulary_context"
        not in text
    )

    assert (
        "BROAD_METHODS_ONLY_CONTEXT_ID"
        not in text
    )

    assert (
        "extraction_adapter.extraction_policy_transform"
        in text
    )

    assert (
        "extraction_adapter.reduced_vocabulary_context_builder"
        in text
    )

    tree = ast.parse(text)

    scientific_imports = set()

    for node in ast.walk(tree):
        modules = []

        if isinstance(node, ast.Import):
            modules = [
                alias.name
                for alias in node.names
            ]

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules = [node.module]

        for module in modules:
            if module.startswith(
                (
                    "domains.dac_her",
                    "domains.sers",
                    "domains.catalysis_mechanism",
                )
            ):
                scientific_imports.add(module)

    assert scientific_imports == set()
