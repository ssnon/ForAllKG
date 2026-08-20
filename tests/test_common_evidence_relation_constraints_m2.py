from __future__ import annotations

import ast
from pathlib import Path

from pipeline_core.evidence_relation_constraints import (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
)

from dac_her.domains.strict_relation_contracts import (
    CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS,
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
        as LEGACY_COMMON,
    DAC_HER_STRICT_RELATION_CONSTRAINTS,
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS,
    SERS_AU_AG_STRICT_RELATION_CONSTRAINTS,
)


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_COMMON = (
    (
        "HAS_MEASUREMENT",
        frozenset({"Experiment", "Calculation"}),
        frozenset({"Measurement"}),
        "warning",
    ),
    (
        "MEASURED_FOR",
        frozenset({"Measurement"}),
        frozenset({"Entity"}),
        "warning",
    ),
    (
        "IN_MEASUREMENT_GROUP",
        frozenset({"Measurement"}),
        frozenset({"MeasurementGroup"}),
        "warning",
    ),
    (
        "SUPPORTS_CLAIM",
        frozenset({
            "Measurement",
            "Experiment",
            "Calculation",
        }),
        frozenset({
            "ObservationClaim",
            "MechanismClaim",
        }),
        "warning",
    ),
    (
        "INTERPRETED_AS",
        frozenset({"ObservationClaim"}),
        frozenset({"MechanismClaim"}),
        "warning",
    ),
    (
        "APPLIES_TO",
        frozenset({
            "ObservationClaim",
            "MechanismClaim",
        }),
        frozenset({"Entity"}),
        "warning",
    ),
)


def _shape(item):
    return (
        item.relation,
        item.source_types,
        item.target_types,
        item.severity,
    )


def test_shared_common_semantics_are_exact():
    assert tuple(
        _shape(item)
        for item
        in COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
    ) == EXPECTED_COMMON


def test_legacy_common_is_canonical_common():
    assert (
        LEGACY_COMMON
        is COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
    )


def test_sers_alias_identity_is_preserved():
    assert (
        SERS_AU_AG_STRICT_RELATION_CONSTRAINTS
        is COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
    )


def test_dac_common_elements_reuse_canonical_objects():
    embedded = (
        DAC_LEGACY_STRICT_RELATION_CONSTRAINTS[5:11]
    )

    assert len(embedded) == len(
        COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
    )

    assert all(
        left is right
        for left, right in zip(
            embedded,
            COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
        )
    )


def test_existing_full_policy_alias_identity_is_preserved():
    assert (
        DAC_HER_STRICT_RELATION_CONSTRAINTS
        is DAC_LEGACY_STRICT_RELATION_CONSTRAINTS
    )

    assert (
        CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS
        is DAC_LEGACY_STRICT_RELATION_CONSTRAINTS
    )


def test_core_constraint_module_has_no_legacy_domain_import():
    path = (
        ROOT
        / "pipeline_core"
        / "evidence_relation_constraints.py"
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
            ):
                violations.append(
                    f"{path}:{node.lineno}:{name}"
                )

    assert violations == []
