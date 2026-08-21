from __future__ import annotations

import ast
from pathlib import Path

import domains.dac_her.extraction as canonical

from pipeline_core.domain.extraction_domain import (
    ExtractionDomainAdapter,
)

from domains.dac_her.relation_constraints import (
    DAC_HER_STRICT_RELATION_CONSTRAINTS,
)


ROOT = Path(__file__).resolve().parents[2]




def test_adapter_uses_shared_core_type():
    assert isinstance(
        canonical.DAC_HER_EXTRACTION_ADAPTER,
        ExtractionDomainAdapter,
    )




def test_adapter_consumes_canonical_relation_policy():
    assert (
        canonical
        .DAC_HER_EXTRACTION_ADAPTER
        .strict_relation_constraints
        is DAC_HER_STRICT_RELATION_CONSTRAINTS
    )


def test_canonical_module_has_no_legacy_dac_import():
    path = (
        ROOT
        / "domains"
        / "dac_her"
        / "extraction.py"
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


def test_registry_imports_canonical_dac_adapter():
    path = (
        ROOT
        / "domains"
        / "extraction_registry.py"
    )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    bindings = []

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue

        for alias in node.names:
            if alias.name == "DAC_HER_EXTRACTION_ADAPTER":
                bindings.append(node.module)

    assert (
        "domains.dac_her.extraction"
        in bindings
    )
