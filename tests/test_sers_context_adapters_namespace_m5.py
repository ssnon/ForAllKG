from __future__ import annotations

import ast
from pathlib import Path

from domains.sers.comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER as CANONICAL_COMPARISON,
    _canonical_concentration as CANONICAL_CONCENTRATION,
    _canonical_entity as CANONICAL_ENTITY,
    _canonical_power as CANONICAL_POWER,
    _canonical_raman_peak as CANONICAL_RAMAN_PEAK,
    _canonical_time as CANONICAL_TIME,
    _canonical_wavelength as CANONICAL_WAVELENGTH,
)

from domains.sers.metric_definition import (
    SERS_AU_AG_METRIC_DEFINITION_ADAPTER as CANONICAL_METRIC,
    _finalize_definition_interpretation as CANONICAL_FINALIZE,
)

from domains.sers.reproducibility import (
    SERS_AU_AG_REPRODUCIBILITY_ADAPTER as CANONICAL_REPRO,
)

from dac_her.domains.sers_au_ag_comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER as LEGACY_COMPARISON,
    _canonical_concentration as LEGACY_CONCENTRATION,
    _canonical_entity as LEGACY_ENTITY,
    _canonical_power as LEGACY_POWER,
    _canonical_raman_peak as LEGACY_RAMAN_PEAK,
    _canonical_time as LEGACY_TIME,
    _canonical_wavelength as LEGACY_WAVELENGTH,
)

from dac_her.domains.sers_au_ag_metric_definition import (
    SERS_AU_AG_METRIC_DEFINITION_ADAPTER as LEGACY_METRIC,
    _finalize_definition_interpretation as LEGACY_FINALIZE,
)

from dac_her.domains.sers_au_ag_reproducibility import (
    SERS_AU_AG_REPRODUCIBILITY_ADAPTER as LEGACY_REPRO,
)

from dac_her.domains.comparison_registry import (
    get_comparison_adapter,
)
from dac_her.domains.metric_definition_registry import (
    get_metric_definition_adapter,
)
from dac_her.domains.reproducibility_registry import (
    get_reproducibility_adapter,
)


ROOT = Path(__file__).resolve().parents[1]


def test_adapter_identity() -> None:
    assert LEGACY_COMPARISON is CANONICAL_COMPARISON
    assert LEGACY_METRIC is CANONICAL_METRIC
    assert LEGACY_REPRO is CANONICAL_REPRO


def test_registry_identity() -> None:
    assert (
        get_comparison_adapter("sers_au_ag")
        is CANONICAL_COMPARISON
    )

    assert (
        get_metric_definition_adapter("sers_au_ag")
        is CANONICAL_METRIC
    )

    assert (
        get_reproducibility_adapter("sers_au_ag")
        is CANONICAL_REPRO
    )


def test_comparison_private_compatibility() -> None:
    pairs = (
        (LEGACY_CONCENTRATION, CANONICAL_CONCENTRATION),
        (LEGACY_ENTITY, CANONICAL_ENTITY),
        (LEGACY_POWER, CANONICAL_POWER),
        (LEGACY_RAMAN_PEAK, CANONICAL_RAMAN_PEAK),
        (LEGACY_TIME, CANONICAL_TIME),
        (LEGACY_WAVELENGTH, CANONICAL_WAVELENGTH),
    )

    for legacy, canonical in pairs:
        assert legacy is canonical


def test_metric_private_compatibility() -> None:
    assert LEGACY_FINALIZE is CANONICAL_FINALIZE


def test_registries_import_canonical_namespaces() -> None:
    expected = {
        (
            "dac_her/domains/comparison_registry.py",
            "domains.sers.comparison",
            "dac_her.domains.sers_au_ag_comparison",
        ),
        (
            "dac_her/domains/metric_definition_registry.py",
            "domains.sers.metric_definition",
            "dac_her.domains.sers_au_ag_metric_definition",
        ),
        (
            "dac_her/domains/reproducibility_registry.py",
            "domains.sers.reproducibility",
            "dac_her.domains.sers_au_ag_reproducibility",
        ),
    }

    for filename, canonical, legacy in expected:
        path = ROOT / filename

        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )

        modules = [
            node.module or ""
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        ]

        assert canonical in modules
        assert legacy not in modules


def test_legacy_modules_are_definition_free() -> None:
    paths = (
        ROOT
        / "dac_her"
        / "domains"
        / "sers_au_ag_comparison.py",
        ROOT
        / "dac_her"
        / "domains"
        / "sers_au_ag_metric_definition.py",
        ROOT
        / "dac_her"
        / "domains"
        / "sers_au_ag_reproducibility.py",
    )

    for path in paths:
        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )

        definitions = [
            node
            for node in tree.body
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            )
        ]

        assert definitions == []


def test_canonical_modules_do_not_reverse_import_legacy() -> None:
    paths = (
        ROOT / "domains" / "sers" / "comparison.py",
        ROOT / "domains" / "sers" / "metric_definition.py",
        ROOT / "domains" / "sers" / "reproducibility.py",
    )

    legacy_prefixes = (
        "dac_her.domains.sers_au_ag_comparison",
        "dac_her.domains.sers_au_ag_metric_definition",
        "dac_her.domains.sers_au_ag_reproducibility",
    )

    for path in paths:
        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
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

        assert not any(
            module.startswith(legacy_prefixes)
            for module in modules
        )
