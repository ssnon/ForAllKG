from __future__ import annotations

import ast
from pathlib import Path

import dac_her.chemistry_signatures as legacy
import pipeline_core.chemistry_signatures as core


ROOT = Path(__file__).resolve().parents[1]


BEHAVIOR_CASES = (
    "",
    "Pt-Ru",
    "W1Mo1",
    "Mo2-NG",
    "FeN4",
    "Au@Ag core-shell",
    "gold and silver nanoparticles",
    "platinum ruthenium catalyst",
    "Commercial material",
    "Nitrogen coordination",
    "STEM XPS CVD analysis",
    "Pt-Ru Pt-Ru repeated",
    "Fe2Co1 catalyst",
)


def test_function_identity_is_preserved():
    assert (
        legacy.composition_signature
        is core.composition_signature
    )

    assert (
        legacy.metal_signature
        is core.metal_signature
    )


def test_constant_identity_is_preserved():
    assert (
        legacy.METAL_SYMBOLS
        is core.METAL_SYMBOLS
    )

    assert (
        legacy.METAL_NAMES
        is core.METAL_NAMES
    )


def test_functions_are_core_owned():
    assert (
        core.composition_signature.__module__
        == "pipeline_core.chemistry_signatures"
    )

    assert (
        core.metal_signature.__module__
        == "pipeline_core.chemistry_signatures"
    )


def test_behavior_is_identical_through_legacy_facade():
    for value in BEHAVIOR_CASES:
        assert (
            legacy.composition_signature(value)
            == core.composition_signature(value)
        )

        assert (
            legacy.metal_signature(value)
            == core.metal_signature(value)
        )


def test_core_has_no_reverse_dependency():
    path = (
        ROOT
        / "pipeline_core"
        / "chemistry_signatures.py"
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


def test_production_consumers_use_core():
    paths = (
        ROOT / "dac_her" / "resolution_candidates.py",
        ROOT / "domains" / "dac_her" / "scientific_signatures.py",
        ROOT / "dac_her" / "semantic_repairs.py",
        ROOT / "dac_her" / "sers_bridge_signatures.py",
    )

    for path in paths:
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
            "pipeline_core.chemistry_signatures"
            in modules
        )

        assert (
            "dac_her.chemistry_signatures"
            not in modules
        )
